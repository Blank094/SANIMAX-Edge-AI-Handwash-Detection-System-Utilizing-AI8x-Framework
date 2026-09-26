# SaniMAX: An Edge AI Real-time Handwash Detection System Utilizing the AI8X Framework

An Edge AI device that classifies handwashing technique in real time entirely on-device — no cloud, no host PC at inference time — built around Analog Devices' ultra-low-power **MAX78000FTHR** CNN accelerator and the **AI8X** model development framework.

> Undergraduate Thesis — Mohammad Naim D. Mariga
> Department of Computer Engineering and Mechatronics, College of Engineering
> MSU-Iligan Institute of Technology, January 2026
> Adviser: Prof. Francis Jann A. Alagon

## Abstract

Proper hand hygiene is one of the simplest, most effective interventions for preventing the spread of infectious disease, yet studies show most people do not wash their hands correctly, and manual compliance monitoring is inconsistent and subjective. SaniMAX addresses this with a self-contained Edge AI device that classifies handwashing gestures against the **WHO handwashing protocol** in real time, using only the compute available on the MAX78000FTHR — avoiding the latency, privacy, and power costs of cloud-based monitoring.

A custom dataset was collected using the MAX78000FTHR's own built-in camera to ensure the training data matched the exact operational input the deployed model would see. Three **AI85Net** CNN variants (64×64, 96×96, and 128×128 input resolution) were trained and evaluated; the 64×64 model was selected for deployment for its optimal accuracy/speed trade-off — **99.471% test accuracy** at **3.3 ms inference latency** (22 FPS system throughput). Model interpretability was validated with Grad-CAM, and the deployed system was evaluated with 15 unseen respondents (96.1% sensitivity) and against out-of-distribution activities (83.5% specificity).

## Motivation

- 95% of people worldwide do not wash their hands correctly, despite handwashing being one of the most cost-effective interventions against infectious disease.
- Manual, human-supervised monitoring is subjective, inconsistent, and offers no real-time feedback.
- Cloud-based vision systems introduce latency, network dependence, privacy concerns, and power costs that make them impractical for continuous, resource-constrained deployment (e.g., healthcare, food processing, laboratories).
- SaniMAX asks: can a fully on-device, ultra-low-power system classify handwashing gestures accurately enough, and fast enough, to give useful real-time feedback?

## System Overview

**Hardware**
- Custom carrier PCB with a push button, 1 red LED (operation indicator), and 3 green LEDs (one per distinct handwashing step completed, regardless of order)
- MAX78000FTHR microcontroller (dual-core Arm Cortex-M4F + RISC-V coprocessor, integrated CNN accelerator, built-in VGA camera) mounted via pin headers
- 3D-printed wall-mounted enclosure, positioned ~1 ft above the sink at a ~140° downward angle

**Software / model pipeline**
1. **Training** — AI85Net CNN trained in PyTorch with Quantization-Aware Training (QAT), Adam optimizer, lr = 0.001, batch size 512, 500 epochs
2. **Synthesis** — Trained/quantized model converted to embedded C via `ai8x-synthesis` (`ai8xize.py`) from a YAML architecture spec
3. **Deployment** — Synthesized inference code + live camera capture deployed on the MAX78000FTHR via MaximSDK

**Feedback logic** — The system classifies gestures continuously in 15-second windows and registers the statistical mode of each window as the user's action for that step. Three 15-second windows (45 s total) align with the WHO's recommended 40–60 second handwash duration; a distinct green LED lights for each of the three unique steps detected, and repeated gestures are ignored until a new one is performed.

## Dataset

- 5 handwashing gestures from the **WHO handwashing protocol**, plus an **"unknown"** class for transitional movements between steps (6 classes total)
- Collected from 32 respondents using the MAX78000FTHR's own camera, at 64×64 / 96×96 / 128×128 RGB888 resolution
- 5,434 raw images, quadrupled to **21,736 images** via horizontal flip, grayscale, and horizontal-flip + grayscale augmentation
- Split 80% train (10% of that held out for validation) / 20% test

| Class | Raw Images |
|---|---|
| Handwash 1 | 930 |
| Handwash 2 | 990 |
| Handwash 3 | 825 |
| Handwash 4 | 940 |
| Handwash 5 | 929 |
| Unknown | 820 |
| **Total** | **5,434** |

## Model

AI85Net CNN — 6 convolutional layers, 1 fully connected layer, softmax output. Three input resolutions were trained and compared:

| Resolution | Top-1 Accuracy | Precision | Recall | F1 | Inference Latency | System Latency | Throughput |
|---|---|---|---|---|---|---|---|
| **64×64** (deployed) | 99.471% | 0.995 | 0.994 | 0.994 | **3.3 ms** | **45 ms** | **~22 FPS** |
| 96×96 | 99.747% | 0.997 | 0.997 | 0.997 | 7.1 ms | 61 ms | ~16 FPS |
| 128×128 | 99.540% | 0.995 | 0.995 | 0.995 | 12.4 ms | 93 ms | ~10 FPS |

All three resolutions achieved near-identical accuracy, so the **64×64 model was selected for deployment** — throughput mattered most for real-time feedback, and the accuracy difference between resolutions was negligible.

Compared against other published models on the same MAX78000 platform, SaniMAX achieved the best combination of speed and accuracy:

| Model | Task | Inference Latency | Accuracy |
|---|---|---|---|
| **SaniMAX (64×64)** | Handwash gesture classification | **3.3 ms** | **99.471%** |
| KWS CNN | Keyword spotting (audio) | 3.5 ms | 96.3% |
| FaceIDNet | Face identification | 28.1 ms | ~84% |

**Interpretability** — Grad-CAM was applied across all five convolutional layers to verify the model's decision-making. It confirmed hierarchical feature learning (edge/texture detection in early layers → gesture-specific semantic understanding in deeper layers) and that the model attends to the correct anatomical cues for each gesture (palm-to-palm contact, interlaced fingers, clasped thumbs, etc.).

## Real-World Evaluation

**Sensitivity** — Tested live with 15 respondents not seen during training, each performing all 5 gestures for 10 seconds:

| Class | Avg. Accuracy |
|---|---|
| Handwash 1 | 97.5% |
| Handwash 2 | 98.7% |
| Handwash 3 | 99.7% |
| Handwash 4 | 92.6% |
| Handwash 5 | 91.8% |
| Unknown | 63.2% |
| **Overall sensitivity** | **96.1%** |

**Specificity (out-of-distribution rejection)** — One respondent performed 5 non-handwashing activities to test false-positive rejection:

| Scenario | Specificity |
|---|---|
| Standing still | 98.2% |
| Right hand only | 100.0% |
| Left hand only | 60.9% |
| Holding a cellphone | 100.0% |
| Opening the faucet | 58.6% |
| **Overall average** | **83.5%** |

The system handled clearly distinct activities (holding a cellphone, right-hand-only) with perfect rejection, but showed lower specificity for activities that visually resemble trained gestures (faucet operation resembling the rotational thumb-rub gesture; left-hand-only resembling the bilateral interlaced-finger gesture) — a direction identified for future data augmentation.

**Feedback mechanism** — Confirmed correct end-to-end behavior in practical testing: three unique gestures each light a distinct green LED regardless of order, and repeated gestures are correctly ignored without triggering a new LED.

## Scientific Contributions

- End-to-end design and validation of a self-contained Edge AI handwash-monitoring system independent of cloud computation
- A device-specific dataset collected directly on the deployment hardware's own camera, eliminating train/deploy domain mismatch
- An empirical accuracy-vs-latency analysis across three input resolutions on the MAX78000 platform
- A Grad-CAM–based interpretability study of the AI85Net architecture's layer-wise feature learning
- A novel 15-second, statistical-mode-based real-time feedback logic aligned with WHO handwashing duration guidelines

## Limitations & Future Work

- Accuracy is sensitive to hand position leaving the fixed camera frame, excessive soap obscuring hand features, and ambient lighting
- Specificity is weaker for activities that visually resemble trained gestures (faucet operation, single-hand movements) — addressable with expanded, more balanced training data
- Recommended next steps: power profiling and sleep/wake management, additional feedback modalities (buzzer/display), multi-sensor fusion (IMU/proximity), temporal models (TCN/3D-CNN) for sequential gesture recognition, real-world deployment trials, data logging/IoT integration, and extension to other hygiene behaviors (PPE, mask compliance)

## Tools & Frameworks

- PyTorch (model training, Quantization-Aware Training)
- [ai8x-training](https://github.com/analogdevicesinc/ai8x-training) & [ai8x-synthesis](https://github.com/analogdevicesinc/ai8x-synthesis) — Analog Devices' MAX78000/MAX78002 toolchain
- MaximSDK (embedded firmware)
- MAX78000FTHR development board
- Grad-CAM (model interpretability)

## Repository Structure

| Stage | Description |
|---|---|
| Training | Dataset preparation, augmentation, and AI85Net training scripts (built on `ai8x-training`) |
| Synthesis | Quantization and C-code generation for the MAX78000 CNN accelerator (`ai8x-synthesis` / `ai8xize.py`) |
| Firmware | Embedded deployment code and LED feedback logic for the MAX78000FTHR (MaximSDK) |
| Hardware | PCB and 3D enclosure design files |

## Acknowledgments

Built on Analog Devices' open-source [AI8X toolchain](https://github.com/analogdevicesinc) for the MAX78000/MAX78002 family of AI microcontrollers. Presented in partial fulfillment of the requirements for the degree of Bachelor of Science in Computer Engineering, MSU-Iligan Institute of Technology.
