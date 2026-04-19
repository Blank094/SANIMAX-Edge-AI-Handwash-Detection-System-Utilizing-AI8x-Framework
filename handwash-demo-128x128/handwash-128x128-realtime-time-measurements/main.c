/******************************************************************************
 *
 * Copyright (C) 2022-2023 Maxim Integrated Products, Inc. (now owned by 
 * Analog Devices, Inc.),
 * Copyright (C) 2023-2024 Analog Devices, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 ******************************************************************************/

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include "mxc.h"
#include "rtc.h"
#include "mxc_device.h"
#include "mxc_sys.h"
#include "fcr_regs.h"
#include "icc.h"
#include "led.h"
#include "tmr.h"
#include "dma.h"
#include "pb.h"
#include "cnn.h"
#include "weights.h"
#include "mxc_delay.h"
#include "camera.h"

#define IMAGE_SIZE_X (128)
#define IMAGE_SIZE_Y (128)

#define CAMERA_FREQ (5 * 1000 * 1000)

char *classes[6] = { "handwash_1", "handwash_2", "handwash_3", "handwash_4", "handwash_5", "   unknown" };

// Classification layer:
static int32_t ml_data[CNN_NUM_OUTPUTS];
static q15_t ml_softmax[CNN_NUM_OUTPUTS];

volatile uint32_t cnn_time; // Stopwatch

static uint32_t input_0[IMAGE_SIZE_X * IMAGE_SIZE_Y]; // buffer for camera image

/* **************************************************************************** */
unsigned int utils_get_time_ms(void)
{
    uint32_t sec, ssec;
    double subsec;
    uint32_t ms;
    MXC_RTC_GetSubSeconds(&ssec);
    subsec = ssec / 4096.0;
    MXC_RTC_GetSeconds(&sec);

    ms = (sec * 1000) + (int)(subsec * 1000);

    return ms;
}

/* **************************************************************************** */
void cnn_load_input(void)
{
    int i;
    const uint32_t *in0 = input_0;
    uint32_t start_time;
    start_time = utils_get_time_ms();

    for (i = 0; i < 16384; i++) {
        // Remove the following line if there is no risk that the source would overrun the FIFO:
        while (((*((volatile uint32_t *)0x50000004) & 1)) != 0) {}
        // Wait for FIFO 0
        *((volatile uint32_t *)0x50000008) = *in0++; // Write FIFO 0
    }

    printf("CNN Load Time: %d ms\n", utils_get_time_ms() - start_time);
}

/* **************************************************************************** */
void capture_process_camera(void)
{
    uint8_t *raw;
    uint32_t imgLen;
    uint32_t w, h;
    int cnt = 0;
    uint8_t r, g, b;
    uint8_t *data = NULL;
    stream_stat_t *stat;
    unsigned int start_time;
    start_time = utils_get_time_ms();

    camera_start_capture_image();

    // Get the details of the image from the camera driver.
    camera_get_image(&raw, &imgLen, &w, &h);
    printf("W:%d H:%d L:%d \n", w, h, imgLen);

    // Get image line by line
    for (int row = 0; row < h; row++) {
        // Wait until camera streaming buffer is full
        while ((data = get_camera_stream_buffer()) == NULL) {
            if (camera_is_image_rcv()) {
                break;
            }
        }

        for (int k = 0; k < 4 * w; k += 4) {
            // data format: 0x00bbggrr
            r = data[k];
            g = data[k + 1];
            b = data[k + 2];
            //skip k+3

            // change the range from [0,255] to [-128,127] and store in buffer for CNN
            input_0[cnt++] = ((b << 16) | (g << 8) | r) ^ 0x00808080;
        }

        // Release stream buffer
        release_camera_stream_buffer();
    }

    //camera_sleep(1);
    stat = get_camera_stream_statistic();

    if (stat->overflow_count > 0) {
        printf("OVERFLOW DISP = %d\n", stat->overflow_count);
        LED_On(LED2); // Turn on red LED if overflow detected
        while (1) {}
    }

    while (camera_is_image_rcv() == 0) {}
    printf("Image Capture and Process Time: %d ms\n", utils_get_time_ms() - start_time);
}

/* **************************************************************************** */
void display_result(void)
{
    int i, digs, tens_val;
    int max_index = 0;

    q15_t max_val = ml_softmax[0];
    for (i = 1; i < CNN_NUM_OUTPUTS; i++) {
        if (ml_softmax[i] > max_val) {
            max_val = ml_softmax[i];
            max_index = i;
        }
    }
    
    // Calculate percentage for only the winning class
    digs = (1000 * max_val + 0x4000) >> 15;
    tens_val = digs % 10;
    digs = digs / 10;

    // Display only the winning class with confidence
    printf("(128x128) Result: %8s (%d.%d%%)\n", classes[max_index], digs, tens_val);
}

/* **************************************************************************** */
int main(void)
{
    int ret = 0;
    int dma_channel;
    unsigned int start_time;
    unsigned int unload_time;

    // Wait for PMIC 1.8V to become available, about 180ms after power up.
    MXC_Delay(200000);
    /* Enable camera power */
    Camera_Power(POWER_ON);
    //MXC_Delay(300000);
    printf("\n\nHandwash Detection Feather Demo\n");

    /* Enable cache */
    MXC_ICC_Enable(MXC_ICC0);

    /* Switch to 100 MHz clock */
    MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO);
    SystemCoreClockUpdate();

    MXC_RTC_Init(0, 0);
    MXC_RTC_Start();

    /* Enable peripheral, enable CNN interrupt, turn on CNN clock */
    /* CNN clock: 50 MHz div 1 */
    cnn_enable(MXC_S_GCR_PCLKDIV_CNNCLKSEL_PCLK, MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV1);

    /* Configure P2.5, turn on the CNN Boost */
    cnn_boost_enable(MXC_GPIO2, MXC_GPIO_PIN_5);

    /* Bring CNN state machine into consistent state */
    cnn_init();
    /* Load CNN kernels */
    cnn_load_weights();
    /* Load CNN bias */
    cnn_load_bias();
    /* Configure CNN state machine */
    cnn_configure();

    // Initialize DMA for camera interface
    MXC_DMA_Init();
    dma_channel = MXC_DMA_AcquireChannel();

    // Initialize camera.
    printf("Init Camera.\n");
    camera_init(CAMERA_FREQ);

    ret = camera_setup(IMAGE_SIZE_X, IMAGE_SIZE_Y, PIXFORMAT_RGB888, FIFO_THREE_BYTE, STREAMING_DMA,
                       dma_channel);
    if (ret != STATUS_OK) {
        printf("Error returned from setting up camera. Error %d\n", ret);
        return -1;
    }

    // For the uart at the top of the camera, we need to flip the image horizontally
    camera_set_hmirror(0);
    camera_set_vflip(0);

    // set the clock speed twice for faster FPS, otherwise set the 0x80 to 0x00 to use the default clock speed
    camera_write_reg(0x11, 0x80); // set camera clock prescaller to prevent streaming overflow

    // Enable CNN clock
    MXC_SYS_ClockEnable(MXC_SYS_PERIPH_CLOCK_CNN);

    while (1) {
        start_time = utils_get_time_ms();
        LED_Toggle(LED1);

        capture_process_camera();

        cnn_start();
        cnn_load_input();

        SCB->SCR &= ~SCB_SCR_SLEEPDEEP_Msk; // SLEEPDEEP=0
        while (cnn_time == 0) {
            __WFI(); // Wait for CNN interrupt
        }

        printf("CNN Time: %d us\n", cnn_time);

        printf("Unload CNN\n\n");
        unload_time = utils_get_time_ms();

        // Unload CNN data
        cnn_unload((uint32_t *)ml_data);
        cnn_stop();

        // Softmax
        softmax_q17p14_q15((const q31_t *)ml_data, CNN_NUM_OUTPUTS, ml_softmax);

        display_result();

        printf("\nUnload/Decision time: %d ms\n", utils_get_time_ms() - unload_time);
        printf("TOTAL TIME: %d ms\n", utils_get_time_ms() - start_time);
        printf("-----------\n\n");
    }

    return 0;
}