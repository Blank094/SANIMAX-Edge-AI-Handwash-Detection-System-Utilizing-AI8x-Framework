#!/usr/bin/env python3
"""
Image to Header File Converter for MAX78000 CNN Handwash Project

This script converts an input image to a C header file format compatible with
the MAX78000 CNN handwash classification project. The output format matches
the SAMPLE_INPUT_0 format used in sampledata.h.

Input: RGB image (any format supported by PIL: PNG, JPG, BMP, etc.)
Output: C header file with image data in HWC format (64x64x3)

Data Format:
- Image is resized to 64x64
- Pixels are converted to signed 8-bit values (-128 to 127) 
- Data is packed as 32-bit words: 0x00BBGGRR per pixel (HWC format)
- Total: 4096 32-bit words (64x64 = 4096 pixels)

Usage:
    python img2header.py <input_image> [output_header]
    python img2header.py --help

Examples:
    python img2header.py test_image.png
    python img2header.py test_image.jpg custom_input.h
"""

import sys
import os
import argparse
from datetime import datetime

try:
    from PIL import Image
    import numpy as np
except ImportError:
    print("Error: This script requires 'Pillow' and 'numpy' packages.")
    print("Install them using: pip install Pillow numpy")
    sys.exit(1)


# Image dimensions expected by the CNN
IMG_WIDTH = 64
IMG_HEIGHT = 64
IMG_CHANNELS = 3

# Class labels for the handwash model (6 classes)
CLASS_NAMES = [
    "Class 0",  # Replace with actual class names if known
    "Class 1",
    "Class 2", 
    "Class 3",
    "Class 4",
    "Class 5"
]


def convert_to_signed(value):
    """
    Convert unsigned 8-bit value (0-255) to signed representation.
    Values are centered around 128, so 128 becomes 0, 0 becomes -128, 255 becomes 127.
    The result is stored as unsigned byte but represents signed value.
    """
    # Convert to signed: subtract 128 to center around 0
    signed_val = value - 128
    # Convert back to unsigned representation for storage
    if signed_val < 0:
        return signed_val + 256  # Two's complement
    return signed_val


def load_and_preprocess_image(image_path):
    """
    Load an image, resize it to 64x64, and convert to RGB format.
    
    Args:
        image_path: Path to the input image file
        
    Returns:
        numpy array of shape (64, 64, 3) with uint8 values
    """
    # Open image
    img = Image.open(image_path)
    
    # Convert to RGB if necessary (handles grayscale, RGBA, etc.)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize to 64x64 using high-quality resampling
    img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    img_array = np.array(img, dtype=np.uint8)
    
    return img_array


def image_to_hwc_data(img_array):
    """
    Convert image array to HWC format data for MAX78000 FIFO input.
    
    The MAX78000 expects data in HWC (Height-Width-Channel) format where
    each pixel is packed as a 32-bit word with RGB channels.
    
    Format: 0x00BBGGRR (Blue in bits 23-16, Green in bits 15-8, Red in bits 7-0)
    
    Note: The sample data shows grayscale-like values where R=G=B, but for
    color images, all three channels are packed together.
    
    Args:
        img_array: numpy array of shape (64, 64, 3)
        
    Returns:
        list of 4096 32-bit unsigned integers
    """
    data = []
    
    for y in range(IMG_HEIGHT):
        for x in range(IMG_WIDTH):
            r = img_array[y, x, 0]
            g = img_array[y, x, 1]
            b = img_array[y, x, 2]
            
            # Convert to signed representation (centered around 128)
            r_signed = convert_to_signed(r)
            g_signed = convert_to_signed(g)
            b_signed = convert_to_signed(b)
            
            # Pack as 0x00BBGGRR
            packed = (b_signed << 16) | (g_signed << 8) | r_signed
            data.append(packed)
    
    return data


def generate_header_content(data, input_filename, macro_name="SAMPLE_INPUT_0"):
    """
    Generate C header file content with the image data.
    
    Args:
        data: list of 32-bit unsigned integers
        input_filename: original image filename for documentation
        macro_name: name of the C macro to define
        
    Returns:
        string containing the header file content
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    header = f"""// This file was @generated automatically by img2header.py
// Source image: {os.path.basename(input_filename)}
// Generated: {timestamp}
// Image dimensions: {IMG_WIDTH}x{IMG_HEIGHT}x{IMG_CHANNELS} (HWC format)
// Total data: {len(data)} 32-bit words ({len(data) * 4} bytes)

#define {macro_name} {{ \\
"""
    
    # Format data as hex values, 8 per line
    lines = []
    for i in range(0, len(data), 8):
        chunk = data[i:i+8]
        hex_values = ", ".join(f"0x{val:08x}" for val in chunk)
        if i + 8 < len(data):
            lines.append(f"  {hex_values}, \\")
        else:
            lines.append(f"  {hex_values} \\")
    
    header += "\n".join(lines)
    header += "\n}\n"
    
    return header


def generate_test_header(input_filename, output_filename):
    """
    Main function to convert an image to a header file.
    
    Args:
        input_filename: path to input image
        output_filename: path to output header file
    """
    print(f"Loading image: {input_filename}")
    
    # Load and preprocess image
    img_array = load_and_preprocess_image(input_filename)
    print(f"Image loaded and resized to {IMG_WIDTH}x{IMG_HEIGHT}")
    
    # Convert to HWC data format
    data = image_to_hwc_data(img_array)
    print(f"Converted to {len(data)} 32-bit words")
    
    # Generate header content
    header_content = generate_header_content(data, input_filename)
    
    # Write to file
    with open(output_filename, 'w') as f:
        f.write(header_content)
    
    print(f"Header file written: {output_filename}")
    print(f"\nTo use this file:")
    print(f"  1. Replace sampledata.h with {os.path.basename(output_filename)}")
    print(f"     or rename {os.path.basename(output_filename)} to sampledata.h")
    print(f"  2. Rebuild the project: make clean && make")
    print(f"  3. Flash and run on MAX78000 board")


def preview_image(input_filename):
    """
    Display image information without converting.
    """
    img = Image.open(input_filename)
    print(f"Image: {input_filename}")
    print(f"  Format: {img.format}")
    print(f"  Mode: {img.mode}")
    print(f"  Size: {img.size[0]}x{img.size[1]}")
    
    if img.mode != 'RGB':
        print(f"  Note: Will be converted to RGB")
    if img.size != (IMG_WIDTH, IMG_HEIGHT):
        print(f"  Note: Will be resized to {IMG_WIDTH}x{IMG_HEIGHT}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert an image to a C header file for MAX78000 CNN handwash classification.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s test_image.png                    # Output: test_image_input.h
  %(prog)s test_image.jpg custom_input.h     # Output: custom_input.h
  %(prog)s --preview test_image.png          # Show image info only

The generated header file can replace sampledata.h to test different images
with the CNN classifier.
        """
    )
    
    parser.add_argument(
        "input_image",
        help="Path to input image file (PNG, JPG, BMP, etc.)"
    )
    
    parser.add_argument(
        "output_header",
        nargs="?",
        default=None,
        help="Path to output header file (default: <input_name>_input.h)"
    )
    
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview image info without converting"
    )
    
    parser.add_argument(
        "--macro",
        default="SAMPLE_INPUT_0",
        help="C macro name for the data (default: SAMPLE_INPUT_0)"
    )
    
    args = parser.parse_args()
    
    # Check input file exists
    if not os.path.isfile(args.input_image):
        print(f"Error: Input file not found: {args.input_image}")
        sys.exit(1)
    
    # Preview mode
    if args.preview:
        preview_image(args.input_image)
        return
    
    # Determine output filename
    if args.output_header:
        output_file = args.output_header
    else:
        base_name = os.path.splitext(os.path.basename(args.input_image))[0]
        output_file = f"{base_name}_input.h"
    
    # Convert image to header
    generate_test_header(args.input_image, output_file)


if __name__ == "__main__":
    main()
