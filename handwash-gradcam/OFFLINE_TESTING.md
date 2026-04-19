# MAX78000 CNN Handwash Classifier - Offline Testing Guide

This guide explains how to test the CNN handwash classifier with custom images using offline testing.

## Project Overview

This project implements a 6-class handwash step classification using the MAX78000's CNN accelerator.

### CNN Architecture
- **Input**: 64×64 RGB image (HWC format)
- **Layers**: 7 layers (6 convolutional + 1 fully connected)
- **Output**: 6 classes (handwash steps)
- **Operations**: ~12.8M ops per inference

### Classes
The model classifies the following handwash steps:
- Class 0: Step 1 - Palm to Palm
- Class 1: Step 2 - Back of Hands
- Class 2: Step 3 - Between Fingers
- Class 3: Step 4 - Back of Fingers
- Class 4: Step 5 - Thumbs
- Class 5: Step 6 - Fingertips

> **Note**: Update the class names in `main.c` to match your actual training dataset.

## Offline Testing Workflow

### Prerequisites
1. Python 3.x installed
2. Required Python packages:
   ```bash
   pip install Pillow numpy
   ```
3. MAX78000 SDK and toolchain configured

### Step 1: Prepare Your Test Image
Your image can be any format supported by PIL (PNG, JPG, BMP, etc.) and any size. The script will:
- Convert to RGB if needed
- Resize to 64×64 using high-quality resampling

### Step 2: Convert Image to Header File

Use the `img2header.py` script to convert your image:

```bash
# Basic usage (output: <imagename>_input.h)
python img2header.py my_test_image.png

# Specify output filename
python img2header.py my_test_image.jpg custom_sampledata.h

# Preview image info without converting
python img2header.py --preview my_test_image.png
```

**Example output:**
```
Loading image: my_test_image.png
Image loaded and resized to 64x64
Converted to 4096 32-bit words
Header file written: my_test_image_input.h

To use this file:
  1. Replace sampledata.h with my_test_image_input.h
     or rename my_test_image_input.h to sampledata.h
  2. Rebuild the project: make clean && make
  3. Flash and run on MAX78000 board
```

### Step 3: Replace Sample Data

**Option A: Replace the file directly**
```bash
# Backup original
copy sampledata.h sampledata.h.bak

# Replace with your converted image
copy my_test_image_input.h sampledata.h
```

**Option B: Modify include in main.c** (for testing multiple images)
```c
// In main.c, change:
#include "sampledata.h"
// To:
#include "my_test_image_input.h"
```

### Step 4: Build the Project

```bash
# Clean and rebuild
make clean
make
```

Or if using Visual Studio Code with the MSDK extension, use the build command.

### Step 5: Flash and Run

1. Connect your MAX78000 board via USB
2. Flash the firmware:
   ```bash
   make flash
   ```
3. Open a serial terminal (115200 baud)
4. Observe the classification results

### Expected Output

```
Waiting...

*** CNN Inference Test handwash ***

*** PASS ***

Approximate data loading and inference time: 1234 us

Classification results:
========================
[  -5024] -> Class 0 (Step 1 - Palm to Palm): 2.5%
[   1234] -> Class 1 (Step 2 - Back of Hands): 85.3% <-- BEST
[  -2048] -> Class 2 (Step 3 - Between Fingers): 5.1%
[  -3000] -> Class 3 (Step 4 - Back of Fingers): 3.2%
[   -512] -> Class 4 (Step 5 - Thumbs): 2.8%
[  -1024] -> Class 5 (Step 6 - Fingertips): 1.1%

========================
PREDICTION: Class 1 - Step 2 - Back of Hands (85.3% confidence)
========================
```

## Data Format Details

### Image Format (HWC)
The MAX78000 CNN expects data in HWC (Height-Width-Channel) format:
- 64 rows × 64 columns × 3 channels = 12,288 bytes
- Each pixel is packed as a 32-bit word: `0x00BBGGRR`
- Pixel values are signed 8-bit (-128 to 127), centered around 128

### Sample Data Structure
The `sampledata.h` file defines:
```c
#define SAMPLE_INPUT_0 { \
  0x008a8a8a, 0x008a8a8a, ... \  // 4096 total 32-bit words
}
```

## Troubleshooting

### Build Errors
- Ensure `sampledata.h` has valid syntax
- Check that the macro is named `SAMPLE_INPUT_0`

### Classification Accuracy
- Use images similar to training data
- Ensure proper lighting and hand positioning
- Check if image orientation matches training data

### "Data mismatch" Error
The `check_output()` function compares against expected output from the original sample.
For testing new images, you may want to comment out this check in `main.c`:
```c
// if (check_output() != CNN_OK) fail();  // Comment this line
softmax_layer();
```

## Files Description

| File | Description |
|------|-------------|
| `img2header.py` | Python script to convert images to header files |
| `main.c` | Main application with CNN inference and classification |
| `sampledata.h` | Sample input image data (replace for testing) |
| `sampleoutput.h` | Expected output for sample data (for verification) |
| `cnn.c` / `cnn.h` | CNN accelerator configuration and control |
| `weights.h` | CNN layer weights |
| `softmax.c` | Softmax function for probability calculation |

## Advanced: Batch Testing

To test multiple images, create a batch script:

```bash
@echo off
REM batch_test.bat - Test multiple images

for %%f in (test_images\*.png) do (
    echo Testing: %%f
    python img2header.py "%%f" sampledata.h
    make clean
    make
    make flash
    echo Press any key for next image...
    pause > nul
)
```

## License

Copyright (C) 2019-2024 Maxim Integrated Products, Inc.
See individual source files for license details.
