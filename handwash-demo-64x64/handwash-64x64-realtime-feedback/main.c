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
 * http://www.apache.org/licenses/LICENSE-2.0
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
#include "gpio.h"
#include "gcr_regs.h" // Needed for the System Reset

#define IMAGE_SIZE_X (64)
#define IMAGE_SIZE_Y (64)

#define CAMERA_FREQ (10 * 1000 * 1000)

#define LED_GREEN_1_PORT    MXC_GPIO0
#define LED_GREEN_1_PIN     MXC_GPIO_PIN_9
#define LED_GREEN_2_PORT    MXC_GPIO0
#define LED_GREEN_2_PIN     MXC_GPIO_PIN_8
#define LED_GREEN_3_PORT    MXC_GPIO0
#define LED_GREEN_3_PIN     MXC_GPIO_PIN_11

#define LED_RED_1_PORT      MXC_GPIO0
#define LED_RED_1_PIN       MXC_GPIO_PIN_6

#define MODE_WINDOW 330     // 15 SECONDS (22FPS)
#define MAX_DISTINCT_CLASSES 3

#define BUTTON_PORT         MXC_GPIO1
#define BUTTON_PIN          MXC_GPIO_PIN_1
#define DEBOUNCE_TIME_MS 150
#define DEBOUNCE_TICKS ((SystemCoreClock / 32 / 1000) * DEBOUNCE_TIME_MS)


const mxc_gpio_cfg_t led_green_1 = {LED_GREEN_1_PORT, LED_GREEN_1_PIN, MXC_GPIO_FUNC_OUT, MXC_GPIO_PAD_NONE, MXC_GPIO_VSSEL_VDDIOH};
const mxc_gpio_cfg_t led_green_2 = {LED_GREEN_2_PORT, LED_GREEN_2_PIN, MXC_GPIO_FUNC_OUT, MXC_GPIO_PAD_NONE, MXC_GPIO_VSSEL_VDDIOH};
const mxc_gpio_cfg_t led_green_3 = {LED_GREEN_3_PORT, LED_GREEN_3_PIN, MXC_GPIO_FUNC_OUT, MXC_GPIO_PAD_NONE, MXC_GPIO_VSSEL_VDDIOH};
const mxc_gpio_cfg_t led_red_1 = {LED_RED_1_PORT, LED_RED_1_PIN, MXC_GPIO_FUNC_OUT, MXC_GPIO_PAD_NONE, MXC_GPIO_VSSEL_VDDIOH};

const mxc_gpio_cfg_t button_cfg = {BUTTON_PORT, BUTTON_PIN, MXC_GPIO_FUNC_IN, MXC_GPIO_PAD_PULL_UP};

int detected_classes[MAX_DISTINCT_CLASSES];
int num_detected = 0;
int pass = 0;

int classification_history[MODE_WINDOW];
int history_index = 0;
int history_count = 0;

char *classes[6] = { "handwash_1", "handwash_2", "handwash_3", "handwash_4", "handwash_5", "   unknown" };

static int32_t ml_data[CNN_NUM_OUTPUTS];
static q15_t ml_softmax[CNN_NUM_OUTPUTS];

volatile uint32_t cnn_time;

uint8_t data565[IMAGE_SIZE_X * 2];

static uint32_t input_0[IMAGE_SIZE_X * IMAGE_SIZE_Y];

volatile int button_pressed = 0;
volatile uint32_t last_button_press_time = 0;

void button_handler(void *cbdata)
{
    uint32_t current_ticks = MXC_TMR_GetCount(MXC_TMR0);
    if ((current_ticks - last_button_press_time) > DEBOUNCE_TICKS) {
        button_pressed = 1;
        last_button_press_time = current_ticks;
    }
}

void setup_button(void)
{
    mxc_tmr_cfg_t tmr_cfg;
    tmr_cfg.pres = TMR_PRES_32;
    tmr_cfg.mode = TMR_MODE_CONTINUOUS;
    tmr_cfg.cmp_cnt = 0;
    tmr_cfg.pol = 0;

    MXC_TMR_Init(MXC_TMR0, &tmr_cfg, false);
    MXC_TMR_Start(MXC_TMR0);

    MXC_GPIO_Config(&button_cfg);
    MXC_GPIO_RegisterCallback(&button_cfg, button_handler, NULL);
    MXC_GPIO_IntConfig(&button_cfg, MXC_GPIO_INT_FALLING);
    MXC_GPIO_EnableInt(button_cfg.port, button_cfg.mask);
    NVIC_EnableIRQ(MXC_GPIO_GET_IRQ(MXC_GPIO_GET_IDX(button_cfg.port)));
}

/* **************************************************************************** */

int compute_mode(int *history, int count, int num_classes) {
    int freq[CNN_NUM_OUTPUTS] = {0};
    int mode = 0;
    int max_freq = 0;

    for (int i = 0; i < count; i++) {
        if (history[i] >= 0 && history[i] < num_classes) {
            freq[history[i]]++;
            if (freq[history[i]] > max_freq) {
                max_freq = freq[history[i]];
                mode = history[i];
            }
        }
    }
    return mode;
}

int is_class_detected(int class_idx) {
    for (int i = 0; i < num_detected; i++) {
        if (detected_classes[i] == class_idx) return 1;
    }
    return 0;
}

void reset_led_state(void) {
    num_detected = 0;
    for (int i = 0; i < MAX_DISTINCT_CLASSES; i++) detected_classes[i] = -1;

    MXC_GPIO_OutClr(led_green_1.port, led_green_1.mask);
    MXC_GPIO_OutClr(led_green_2.port, led_green_2.mask);
    MXC_GPIO_OutClr(led_green_3.port, led_green_3.mask);

    pass = 0;
}

/* **************************************************************************** */
void setup_feedback_leds(void)
{
    MXC_GPIO_Config(&led_green_1);
    MXC_GPIO_Config(&led_green_2);
    MXC_GPIO_Config(&led_green_3);
    MXC_GPIO_Config(&led_red_1);
}

void toggle_operation_indicator(void) {
    static int led_state = 0;

    if (led_state) {
        MXC_GPIO_OutClr(led_red_1.port, led_red_1.mask);
        led_state = 0;
    } else {
        MXC_GPIO_OutSet(led_red_1.port, led_red_1.mask);
        led_state = 1;
    }
}

/* **************************************************************************** */
int cnn_display_decision(void)
{
    int32_t max = ml_data[0];
    int32_t max_index = 0;

    for (int i = 1; i < CNN_NUM_OUTPUTS; i++) {
        if (ml_data[i] > max) {
            max = ml_data[i];
            max_index = i;
        }
    }
    return max_index;
}

/* **************************************************************************** */
void cnn_load_input(void)
{
    const uint32_t *in0 = input_0;

    for (int i = 0; i < 4096; i++) {
        while (((*((volatile uint32_t *)0x50000004) & 1)) != 0) {}
        *((volatile uint32_t *)0x50000008) = *in0++;
    }
}

/* **************************************************************************** */
void capture_process_camera(void)
{
    uint8_t *raw;
    uint32_t imgLen;
    uint32_t w, h;
    int cnt = 0;
    uint8_t r, g, b;
    uint16_t rgb;
    int j = 0;
    uint8_t *data = NULL;
    stream_stat_t *stat;

    camera_start_capture_image();
    camera_get_image(&raw, &imgLen, &w, &h);

    for (int row = 0; row < h; row++) {
        while ((data = get_camera_stream_buffer()) == NULL) {
            if (camera_is_image_rcv()) {
                break;
            }
        }
        j = 0;
        for (int k = 0; k < 4 * w; k += 4) {
            r = data[k];
            g = data[k + 1];
            b = data[k + 2];
            input_0[cnt++] = ((b << 16) | (g << 8) | r) ^ 0x00808080;
            rgb = ((r & 0b11111000) << 8) | ((g & 0b11111100) << 3) | (b >> 3);
            data565[j] = (rgb >> 8) & 0xFF;
            data565[j + 1] = rgb & 0xFF;
            j += 2;
        }
        release_camera_stream_buffer();
    }

    stat = get_camera_stream_statistic();
    if (stat->overflow_count > 0) {
        printf("OVERFLOW DISP = %d\n", stat->overflow_count);
    }
}


/* **************************************************************************** */
int main(void)
{
    int dma_channel;

    MXC_Delay(200000);
    Camera_Power(POWER_ON);
    printf("\n\nHandwash Detection Feather Demo\n");

    MXC_ICC_Enable(MXC_ICC0);
    MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO);
    SystemCoreClockUpdate();

    cnn_enable(MXC_S_GCR_PCLKDIV_CNNCLKSEL_PCLK, MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV1);
    cnn_boost_enable(MXC_GPIO2, MXC_GPIO_PIN_5);
    cnn_init();
    cnn_load_weights();
    cnn_load_bias();
    cnn_configure();

    MXC_DMA_Init();
    dma_channel = MXC_DMA_AcquireChannel();

    printf("Init Camera.\n");
    camera_init(CAMERA_FREQ);
    camera_setup(IMAGE_SIZE_X, IMAGE_SIZE_Y, PIXFORMAT_RGB888, FIFO_THREE_BYTE, STREAMING_DMA, dma_channel);
    camera_set_hmirror(0);
    camera_set_vflip(0);
    camera_write_reg(0x11, 0x80);

    MXC_SYS_ClockEnable(MXC_SYS_PERIPH_CLOCK_CNN);

    setup_feedback_leds();
    setup_button();

    for (int i = 0; i < MAX_DISTINCT_CLASSES; i++) detected_classes[i] = -1;
    
    printf("Press the button to start detection...\n");
    while(!button_pressed) {
        // Wait for the first button press to start
    }
    button_pressed = 0; // Consume the button press

    printf("Get ready! Starting detection...\n");
    MXC_Delay(MXC_DELAY_MSEC(3000));
    
    printf("System activated. Press button again to reset.\n");
    reset_led_state();


    while (1) {
        // toggle_operation_indicator();

        // // Testing
        // MXC_GPIO_OutClr(led_green_1.port, led_green_1.mask);
        // MXC_GPIO_OutClr(led_green_2.port, led_green_2.mask);
        // MXC_GPIO_OutClr(led_green_3.port, led_green_3.mask);
        // MXC_GPIO_OutClr(led_red_1.port, led_red_1.mask);
        // MXC_Delay(MXC_DELAY_MSEC(2000));
        // MXC_GPIO_OutSet(led_green_1.port, led_green_1.mask);
        // MXC_GPIO_OutSet(led_green_2.port, led_green_2.mask);
        // MXC_GPIO_OutSet(led_green_3.port, led_green_3.mask);
        // MXC_GPIO_OutSet(led_red_1.port, led_red_1.mask);
        // MXC_Delay(MXC_DELAY_MSEC(2000));

        if (button_pressed) {
            printf("Resetting system...\n");
            MXC_Delay(100000); 
            
            // This line triggers a software reset, rebooting the microcontroller
            MXC_GCR->rst0 = MXC_F_GCR_RST0_SYS;
        }

        toggle_operation_indicator();
        capture_process_camera();
        
        cnn_time = 0;
        cnn_start();
        cnn_load_input();
        
        while (cnn_time == 0) {
            __WFI();
        }
        
        cnn_unload((uint32_t *)ml_data);
        softmax_q17p14_q15((const q31_t *)ml_data, CNN_NUM_OUTPUTS, ml_softmax);

        int decision = cnn_display_decision();
        printf("  Decision : %8s\n", classes[decision]);

        classification_history[history_index++] = decision;
        if (history_index >= MODE_WINDOW) history_index = 0;
        if (history_count < MODE_WINDOW) history_count++;

        if (history_count == MODE_WINDOW) {
            int mode = compute_mode(classification_history, MODE_WINDOW, CNN_NUM_OUTPUTS);
            printf("MODE of last %d: %s\n", MODE_WINDOW, classes[mode]);

            if (!is_class_detected(mode) && mode < 5) {
                detected_classes[num_detected++] = mode;
                if (num_detected == 1) {
                    printf("GREEN LED 1 ON\n");
                    MXC_GPIO_OutSet(led_green_1.port, led_green_1.mask);
                    MXC_Delay(MXC_DELAY_MSEC(2000));
                } else if (num_detected == 2) {
                    printf("GREEN LED 2 ON\n");
                    MXC_GPIO_OutSet(led_green_2.port, led_green_2.mask);
                    MXC_Delay(MXC_DELAY_MSEC(2000));
                } else if (num_detected >= 3) {
                    printf("GREEN LED 3 ON\n");
                    MXC_GPIO_OutSet(led_green_3.port, led_green_3.mask);
                    pass = 1;
                }
            } else {
                printf("Unknown class or repeated step detected - ignore\n");
            }

            if (pass) {
                printf("COMPLIANCE!\n");
                for (int i = 0; i < 3; i++) {
                    MXC_GPIO_OutClr(led_green_1.port, led_green_1.mask);
                    MXC_GPIO_OutClr(led_green_2.port, led_green_2.mask);
                    MXC_GPIO_OutClr(led_green_3.port, led_green_3.mask);
                    MXC_Delay(MXC_DELAY_MSEC(500));
                    MXC_GPIO_OutSet(led_green_1.port, led_green_1.mask);
                    MXC_GPIO_OutSet(led_green_2.port, led_green_2.mask);
                    MXC_GPIO_OutSet(led_green_3.port, led_green_3.mask);
                    MXC_Delay(MXC_DELAY_MSEC(500));
                }
                MXC_Delay(MXC_DELAY_MSEC(2000));
                reset_led_state();
            }

            history_count = 0;
            history_index = 0;
        }
    }

    return 0;
}