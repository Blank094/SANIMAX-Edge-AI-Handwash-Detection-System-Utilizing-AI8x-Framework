#!/usr/bin/env python3
"""
Combined Image Processing Tool for MAX78000 CNN Handwash Classification

This script performs:
1. Converts an image to C header file for MAX78000 CNN inference
2. Generates Grad-CAM visualization showing model attention
3. Outputs both the header file and Grad-CAM PNG

Usage:
    python process_image.py <input_image> [options]

Examples:
    python process_image.py test_image.png
    python process_image.py test_image.jpg --checkpoint trained/model.pth.tar
"""

import sys
import os
import argparse
from datetime import datetime

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
except ImportError as e:
    print(f"Error: Missing required package - {e}")
    print("Install required packages using:")
    print("  pip install torch torchvision numpy Pillow matplotlib")
    sys.exit(1)


# ============================================================================
# Configuration
# ============================================================================

IMG_WIDTH = 64
IMG_HEIGHT = 64
IMG_CHANNELS = 3
NUM_CLASSES = 6

CLASS_NAMES = [
    "Handwash 1",
    "Handwash 2",
    "Handwash 3",
    "Handwash 4",
    "Handwash 5",
    "Unknown"
]


# ============================================================================
# Image to Header Conversion (from img2header.py)
# ============================================================================

def convert_to_signed(value):
    """Convert unsigned 8-bit value (0-255) to signed representation."""
    signed_val = value - 128
    if signed_val < 0:
        return signed_val + 256
    return signed_val


def image_to_hwc_data(img_array):
    """Convert image array to HWC format data for MAX78000 FIFO input."""
    data = []
    for y in range(IMG_HEIGHT):
        for x in range(IMG_WIDTH):
            r = img_array[y, x, 0]
            g = img_array[y, x, 1]
            b = img_array[y, x, 2]
            
            r_signed = convert_to_signed(r)
            g_signed = convert_to_signed(g)
            b_signed = convert_to_signed(b)
            
            packed = (b_signed << 16) | (g_signed << 8) | r_signed
            data.append(packed)
    return data


def generate_header_file(img_array, input_filename, output_filename):
    """Generate C header file with image data."""
    data = image_to_hwc_data(img_array)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    header = f"""// This file was @generated automatically by process_image.py
// Source image: {os.path.basename(input_filename)}
// Generated: {timestamp}
// Image dimensions: {IMG_WIDTH}x{IMG_HEIGHT}x{IMG_CHANNELS} (HWC format)
// Total data: {len(data)} 32-bit words ({len(data) * 4} bytes)

#define SAMPLE_INPUT_0 {{ \\
"""
    
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
    
    with open(output_filename, 'w') as f:
        f.write(header)
    
    return output_filename


# ============================================================================
# CNN Model (matching MAX78000 architecture)
# ============================================================================

class HandwashCNN(nn.Module):
    """PyTorch model matching the MAX78000 CNN architecture.
    Note: No BatchNorm - ai8x uses quantized activations instead."""
    
    def __init__(self, num_classes=NUM_CLASSES):
        super(HandwashCNN, self).__init__()
        
        self.conv0 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.conv5 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.fc = nn.Linear(16 * 4 * 4, num_classes, bias=True)
        
        # Grad-CAM storage - activations captured AFTER ReLU
        self.gradients = None
        self.activations = None
        self.target_layer = 'layer3'  # default
        
    def forward(self, x):
        # Layer 0: 64x64 -> 64x64
        x = F.relu(self.conv0(x))
        if self.target_layer == 'layer0':
            x.register_hook(self._save_gradient_hook)
            self.activations = x
        
        # Layer 1: 64x64 -> 32x32
        x = self.pool1(x)
        x = F.relu(self.conv1(x))
        if self.target_layer == 'layer1':
            x.register_hook(self._save_gradient_hook)
            self.activations = x
        
        # Layer 2: 32x32 -> 16x16
        x = self.pool2(x)
        x = F.relu(self.conv2(x))
        if self.target_layer == 'layer2':
            x.register_hook(self._save_gradient_hook)
            self.activations = x
        
        # Layer 3: 16x16 -> 8x8 (good for Grad-CAM)
        x = self.pool3(x)
        x = F.relu(self.conv3(x))
        if self.target_layer == 'layer3':
            x.register_hook(self._save_gradient_hook)
            self.activations = x
        
        # Layer 4: 8x8 -> 4x4
        x = self.pool4(x)
        x = F.relu(self.conv4(x))
        if self.target_layer == 'layer4':
            x.register_hook(self._save_gradient_hook)
            self.activations = x
        
        # Layer 5: 4x4 -> 4x4
        x = F.relu(self.conv5(x))
        if self.target_layer == 'layer5':
            x.register_hook(self._save_gradient_hook)
            self.activations = x
        
        # Layer 6 - Flatten and FC
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
    
    def _save_gradient_hook(self, grad):
        """Hook to capture gradients during backward pass"""
        self.gradients = grad
    
    def set_target_layer(self, layer_name):
        """Set which layer to use for Grad-CAM."""
        valid_layers = ['layer0', 'layer1', 'layer2', 'layer3', 'layer4', 'layer5']
        if layer_name not in valid_layers:
            print(f"Warning: Invalid layer {layer_name}, using layer3")
            layer_name = 'layer3'
        self.target_layer = layer_name
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self):
        return self.activations


# ============================================================================
# Grad-CAM Implementation
# ============================================================================

class GradCAM:
    def __init__(self, model, target_layer='layer3'):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        # Set the target layer for activation capture
        self.model.set_target_layer(target_layer)
        
    def generate(self, input_tensor, target_class=None):
        # Enable gradients for input
        input_tensor = input_tensor.clone().requires_grad_(True)
        
        # Forward pass
        output = self.model(input_tensor)
        class_scores = output.detach().cpu().numpy()[0]
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        if input_tensor.grad is not None:
            input_tensor.grad.zero_()
        
        # Backward pass for target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)
        
        # Get gradients and activations from hooks
        gradients = self.model.get_activations_gradient()
        activations = self.model.get_activations()
        
        if gradients is None or activations is None:
            print("Warning: Could not capture gradients/activations")
            return np.ones((8, 8)) * 0.5, target_class, class_scores
        
        # Global average pooling of gradients to get channel weights
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        heatmap = cam.squeeze().detach().cpu().numpy()
        return heatmap, target_class, class_scores


# ============================================================================
# Visualization
# ============================================================================

def create_gradcam_overlay(original_img, heatmap, alpha=0.5):
    """Create Grad-CAM overlay on original image."""
    # Upscale heatmap to image size
    heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
    heatmap_resized = np.array(heatmap_img.resize(
        (IMG_WIDTH * 4, IMG_HEIGHT * 4), Image.Resampling.BILINEAR)) / 255.0
    
    # Apply colormap
    colormap = plt.colormaps['jet']
    heatmap_colored = (colormap(heatmap_resized)[:, :, :3] * 255).astype(np.uint8)
    
    # Resize original
    original_resized = np.array(original_img.resize(
        (IMG_WIDTH * 4, IMG_HEIGHT * 4), Image.Resampling.LANCZOS))
    
    # Blend
    overlay = (original_resized * (1 - alpha) + heatmap_colored * alpha).astype(np.uint8)
    return Image.fromarray(overlay)


def create_full_visualization(original_img, heatmap, predicted_class, class_scores, 
                              output_path, image_name):
    """Create comprehensive visualization figure."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Grad-CAM Analysis: {image_name}', fontsize=16, fontweight='bold')
    
    # Original image
    original_resized = original_img.resize((IMG_WIDTH * 4, IMG_HEIGHT * 4), 
                                            Image.Resampling.LANCZOS)
    axes[0, 0].imshow(original_resized)
    axes[0, 0].set_title('Original Image (64x64 input)', fontsize=12)
    axes[0, 0].axis('off')
    
    # Heatmap
    heatmap_upscaled = np.array(Image.fromarray(
        (heatmap * 255).astype(np.uint8)
    ).resize((IMG_WIDTH * 4, IMG_HEIGHT * 4), Image.Resampling.BILINEAR)) / 255.0
    
    im = axes[0, 1].imshow(heatmap_upscaled, cmap='jet')
    axes[0, 1].set_title('Grad-CAM Heatmap', fontsize=12)
    axes[0, 1].axis('off')
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # Overlay
    overlay = create_gradcam_overlay(original_img, heatmap, alpha=0.5)
    axes[0, 2].imshow(overlay)
    axes[0, 2].set_title('Grad-CAM Overlay', fontsize=12)
    axes[0, 2].axis('off')
    
    # Raw heatmap
    axes[1, 0].imshow(heatmap, cmap='jet', interpolation='nearest')
    axes[1, 0].set_title(f'Raw Activation Map ({heatmap.shape[0]}x{heatmap.shape[1]})', fontsize=12)
    axes[1, 0].axis('off')
    
    # Probabilities
    exp_scores = np.exp(class_scores - np.max(class_scores))
    probabilities = exp_scores / np.sum(exp_scores) * 100
    
    colors = ['#2ecc71' if i == predicted_class else '#3498db' for i in range(NUM_CLASSES)]
    bars = axes[1, 1].barh(range(NUM_CLASSES), probabilities, color=colors)
    axes[1, 1].set_yticks(range(NUM_CLASSES))
    axes[1, 1].set_yticklabels(CLASS_NAMES, fontsize=10)
    axes[1, 1].set_xlabel('Confidence (%)', fontsize=11)
    axes[1, 1].set_title('Classification Results', fontsize=12)
    axes[1, 1].set_xlim(0, 105)
    
    for bar, prob in zip(bars, probabilities):
        axes[1, 1].text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                        f'{prob:.1f}%', va='center', fontsize=9)
    
    # Summary
    axes[1, 2].axis('off')
    summary = f"""
╔══════════════════════════════════════╗
║         CLASSIFICATION RESULT        ║
╠══════════════════════════════════════╣
║                                      ║
║  Predicted: Class {predicted_class}                 ║
║  Name: {CLASS_NAMES[predicted_class]:<24} ║
║  Confidence: {probabilities[predicted_class]:>5.1f}%                 ║
║                                      ║
╠══════════════════════════════════════╣
║  The heatmap shows regions the       ║
║  model focuses on for prediction.    ║
║  Warmer colors = higher attention    ║
╚══════════════════════════════════════╝
"""
    axes[1, 2].text(0.05, 0.95, summary, transform=axes[1, 2].transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return output_path


# ============================================================================
# Main Processing
# ============================================================================

def load_and_process_image(image_path):
    """Load image and prepare for both header generation and Grad-CAM."""
    img = Image.open(image_path)
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    original_img = img.copy()
    img_resized = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.LANCZOS)
    img_array = np.array(img_resized, dtype=np.uint8)
    
    # For PyTorch model
    img_normalized = (img_array.astype(np.float32) - 128.0) / 128.0
    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)
    
    return img_array, img_tensor, original_img


def main():
    parser = argparse.ArgumentParser(
        description="Process image for MAX78000 CNN: generate header file and Grad-CAM visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("input_image", help="Path to input image file")
    parser.add_argument("--output", "-o", default=None, 
                        help="Output path for Grad-CAM visualization (default: <input>_gradcam_analysis.png)")
    parser.add_argument("--output-dir", "-d", default=".", help="Output directory (used if -o not specified)")
    parser.add_argument("--checkpoint", "-c", help="Path to PyTorch checkpoint file")
    parser.add_argument("--layer", type=str, default="layer3",
                        choices=["layer0", "layer1", "layer2", "layer3", "layer4", "layer5"],
                        help="Target layer for Grad-CAM visualization (default: layer3). "
                             "Activations captured AFTER ReLU. "
                             "layer0=64x64, layer1=32x32, layer2=16x16, layer3=8x8, layer4/5=4x4.")
    parser.add_argument("--no-header", action="store_true", help="Skip header file generation")
    parser.add_argument("--no-gradcam", action="store_true", help="Skip Grad-CAM generation")
    parser.add_argument("--replace-sampledata", action="store_true", 
                        help="Replace sampledata.h directly")
    
    args = parser.parse_args()
    
    if not os.path.isfile(args.input_image):
        print(f"Error: Input file not found: {args.input_image}")
        sys.exit(1)
    
    base_name = os.path.splitext(os.path.basename(args.input_image))[0]
    
    print(f"\n{'='*60}")
    print(f"Processing: {args.input_image}")
    print(f"{'='*60}\n")
    
    # Load and process image
    img_array, img_tensor, original_img = load_and_process_image(args.input_image)
    print(f"✓ Image loaded and resized to {IMG_WIDTH}x{IMG_HEIGHT}")
    
    # Generate header file
    if not args.no_header:
        if args.replace_sampledata:
            header_path = os.path.join(args.output_dir, "sampledata.h")
        else:
            header_path = os.path.join(args.output_dir, f"{base_name}_input.h")
        
        generate_header_file(img_array, args.input_image, header_path)
        print(f"✓ Header file generated: {header_path}")
    
    # Generate Grad-CAM
    if not args.no_gradcam:
        print("✓ Initializing CNN model...")
        model = HandwashCNN(num_classes=NUM_CLASSES)
        
        if args.checkpoint and os.path.isfile(args.checkpoint):
            try:
                checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
                state_dict = checkpoint.get('state_dict', checkpoint)
                
                # Map ai8x layer names to our model names
                weight_mapping = {
                    'conv1.op.weight': 'conv0.weight',
                    'conv2.op.weight': 'conv1.weight',
                    'conv3.op.weight': 'conv2.weight',
                    'conv4.op.weight': 'conv3.weight',
                    'conv5.op.weight': 'conv4.weight',
                    'conv6.op.weight': 'conv5.weight',
                    'fc.op.weight': 'fc.weight',
                    'fc.op.bias': 'fc.bias',
                }
                
                new_state_dict = {}
                for ai8x_name, our_name in weight_mapping.items():
                    if ai8x_name in state_dict:
                        new_state_dict[our_name] = state_dict[ai8x_name]
                
                model.load_state_dict(new_state_dict, strict=False)
                model.eval()
                
                # Disable batch norm tracking since we don't have running stats
                for module in model.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        module.track_running_stats = False
                        module.running_mean = None
                        module.running_var = None
                
                print(f"✓ Loaded checkpoint: {args.checkpoint}")
            except Exception as e:
                print(f"⚠ Could not load checkpoint: {e}")
                print("  Using random weights for visualization")
        else:
            print("⚠ No checkpoint provided - using random weights")
            print("  (Provide --checkpoint for accurate Grad-CAM)")
        
        # Generate Grad-CAM
        print(f"✓ Generating Grad-CAM (target layer: {args.layer})...")
        gradcam = GradCAM(model, target_layer=args.layer)
        heatmap, predicted_class, class_scores = gradcam.generate(img_tensor)
        
        # Calculate probabilities
        exp_scores = np.exp(class_scores - np.max(class_scores))
        probabilities = exp_scores / np.sum(exp_scores) * 100
        
        # Determine output paths
        if args.output:
            analysis_path = args.output
            overlay_path = os.path.splitext(args.output)[0] + "_overlay.png"
        else:
            overlay_path = os.path.join(args.output_dir, f"{base_name}_gradcam_overlay.png")
            analysis_path = os.path.join(args.output_dir, f"{base_name}_gradcam_analysis.png")
        
        # Save overlay image
        overlay = create_gradcam_overlay(original_img, heatmap, alpha=0.5)
        overlay.save(overlay_path)
        print(f"✓ Grad-CAM overlay saved: {overlay_path}")
        
        # Save full analysis figure
        create_full_visualization(
            original_img, heatmap, predicted_class, class_scores,
            analysis_path, os.path.basename(args.input_image)
        )
        print(f"✓ Full analysis saved: {analysis_path}")
        
        # Print results
        print(f"\n{'='*60}")
        print("CLASSIFICATION RESULT")
        print(f"{'='*60}")
        print(f"Predicted Class: {predicted_class} ({CLASS_NAMES[predicted_class]})")
        print(f"Confidence: {probabilities[predicted_class]:.1f}%")
        print(f"{'='*60}")
        print("\nAll class probabilities:")
        for i, (name, prob) in enumerate(zip(CLASS_NAMES, probabilities)):
            marker = " ◄ BEST" if i == predicted_class else ""
            print(f"  [{i}] {name}: {prob:.1f}%{marker}")
    
    print(f"\n{'='*60}")
    print("Processing complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
