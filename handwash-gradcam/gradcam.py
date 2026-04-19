#!/usr/bin/env python3
"""
Grad-CAM Visualization for MAX78000 CNN Handwash Classification

This script generates Grad-CAM (Gradient-weighted Class Activation Mapping)
visualizations for the handwash classification model. It shows which regions
of the input image the model focuses on when making predictions.

Usage:
    python gradcam.py <input_image> [options]
    python gradcam.py --help

Examples:
    python gradcam.py test_image.png
    python gradcam.py test_image.jpg --output gradcam_result.png
    python gradcam.py test_image.png --checkpoint path/to/model.pth.tar
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


# Image dimensions expected by the CNN
IMG_WIDTH = 64
IMG_HEIGHT = 64
IMG_CHANNELS = 3
NUM_CLASSES = 6

# Class labels
CLASS_NAMES = [
    "Handwash 1",
    "Handwash 2",
    "Handwash 3",
    "Handwash 4",
    "Handwash 5",
    "Unknown"
]


class HandwashCNN(nn.Module):
    """
    PyTorch model matching the MAX78000 CNN architecture for handwash classification.
    
    Architecture (from cnn.c):
    - Layer 0: 3x64x64 -> conv2d 3x3, ReLU -> 16x64x64
    - Layer 1: 16x64x64 -> maxpool 2x2, conv2d 3x3, ReLU -> 32x32x32
    - Layer 2: 32x32x32 -> maxpool 2x2, conv2d 3x3, ReLU -> 64x16x16
    - Layer 3: 64x16x16 -> maxpool 2x2, conv2d 3x3, ReLU -> 32x8x8
    - Layer 4: 32x8x8 -> maxpool 2x2, conv2d 3x3, ReLU -> 32x4x4
    - Layer 5: 32x4x4 -> conv2d 3x3, ReLU -> 16x4x4
    - Layer 6: 16x4x4 (256) -> linear -> 6
    
    Note: No BatchNorm - ai8x uses quantized activations instead
    """
    
    def __init__(self, num_classes=NUM_CLASSES):
        super(HandwashCNN, self).__init__()
        
        # Layer 0: 3 -> 16 (64x64 output)
        self.conv0 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Layer 1: 16 -> 32 with maxpool (32x32 output)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Layer 2: 32 -> 64 with maxpool (16x16 output)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Layer 3: 64 -> 32 with maxpool (8x8 output)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Layer 4: 32 -> 32 with maxpool (4x4 output)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Layer 5: 32 -> 16 (4x4 output)
        self.conv5 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Layer 6: Fully connected 256 -> 6
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
        """Set which layer to use for Grad-CAM.
        
        Args:
            layer_name: 'layer0' (64x64), 'layer1' (32x32), 'layer2' (16x16),
                       'layer3' (8x8), 'layer4' (4x4), 'layer5' (4x4)
        """
        valid_layers = ['layer0', 'layer1', 'layer2', 'layer3', 'layer4', 'layer5']
        if layer_name not in valid_layers:
            print(f"Warning: Invalid layer {layer_name}, using layer3")
            layer_name = 'layer3'
        self.target_layer = layer_name
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self):
        return self.activations


class GradCAM:
    """Grad-CAM implementation for CNN visualization"""
    
    def __init__(self, model, target_layer='layer3'):
        """
        Initialize Grad-CAM with model and target layer.
        
        Args:
            model: The CNN model
            target_layer: Layer to use for visualization (activations captured AFTER ReLU)
                         'layer0' = 64x64 resolution (very detailed, low-level features)
                         'layer1' = 32x32 resolution (detailed)
                         'layer2' = 16x16 resolution (good balance)
                         'layer3' = 8x8 resolution (recommended, semantic features)
                         'layer4' = 4x4 resolution (high-level features)
                         'layer5' = 4x4 resolution (most abstract)
        """
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        # Set the target layer for activation capture
        self.model.set_target_layer(target_layer)
        
    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap for the input image.
        
        Args:
            input_tensor: Preprocessed input tensor (1, 3, 64, 64)
            target_class: Class index to visualize (None = use predicted class)
            
        Returns:
            heatmap: numpy array of shape (H, W) with values 0-1
            predicted_class: The class index predicted by the model
            class_scores: Raw output scores for all classes
        """
        # Enable gradients for input
        input_tensor = input_tensor.clone().requires_grad_(True)
        
        # Forward pass
        output = self.model(input_tensor)
        class_scores = output.detach().cpu().numpy()[0]
        
        # Get predicted class if not specified
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
            # Return uniform heatmap as fallback
            return np.ones((8, 8)) * 0.5, target_class, class_scores
        
        # Global average pooling of gradients to get channel weights
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # ReLU to keep only positive contributions
        cam = F.relu(cam)
        
        # Normalize to 0-1
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        # Convert to numpy and resize to input dimensions
        heatmap = cam.squeeze().detach().cpu().numpy()
        
        return heatmap, target_class, class_scores


def load_and_preprocess_image(image_path):
    """Load and preprocess image for the model"""
    img = Image.open(image_path)
    
    # Convert to RGB
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Store original for visualization
    original_img = img.copy()
    
    # Resize to model input size
    img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.LANCZOS)
    
    # Convert to numpy and normalize to [-1, 1] (matching ai8x training)
    img_array = np.array(img, dtype=np.float32)
    img_array = (img_array - 128.0) / 128.0  # Normalize to [-1, 1]
    
    # Convert to tensor (NCHW format)
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    
    return img_tensor, original_img


def create_gradcam_visualization(original_img, heatmap, alpha=0.5):
    """
    Overlay Grad-CAM heatmap on the original image.
    
    Args:
        original_img: PIL Image (original size)
        heatmap: numpy array (model output size, e.g., 4x4)
        alpha: Blend factor for heatmap overlay
        
    Returns:
        PIL Image with Grad-CAM overlay
    """
    # Resize heatmap to original image size
    heatmap_resized = np.array(Image.fromarray(
        (heatmap * 255).astype(np.uint8)
    ).resize(original_img.size, Image.Resampling.BILINEAR)) / 255.0
    
    # Apply colormap (jet)
    colormap = plt.colormaps['jet']
    heatmap_colored = colormap(heatmap_resized)[:, :, :3]  # Remove alpha channel
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    
    # Convert original image to numpy
    original_array = np.array(original_img.resize((IMG_WIDTH * 4, IMG_HEIGHT * 4), 
                                                   Image.Resampling.LANCZOS))
    
    # Resize heatmap to match
    heatmap_colored = np.array(Image.fromarray(heatmap_colored).resize(
        (IMG_WIDTH * 4, IMG_HEIGHT * 4), Image.Resampling.BILINEAR))
    
    # Blend images
    overlay = (original_array * (1 - alpha) + heatmap_colored * alpha).astype(np.uint8)
    
    return Image.fromarray(overlay), Image.fromarray(heatmap_colored)


def create_visualization_figure(original_img, heatmap, predicted_class, class_scores,
                                 output_path, image_name):
    """
    Create a portrait-oriented visualization figure suitable for IEEE format papers.

    Layout (3 rows x 2 cols with bottom row spanning full width):
      Row 0: (a) Original Image  |  (b) Grad-CAM Heatmap
      Row 1: (c) Grad-CAM Overlay  |  (d) Raw Heatmap
      Row 2: (e) Classification Probabilities  (full width)

    Output: 7 in wide x 9 in tall @ 300 dpi — fits IEEE double-column figure width.
    """
    from matplotlib.gridspec import GridSpec

    # IEEE-friendly base font size
    plt.rcParams.update({'font.family': 'serif', 'font.size': 8})

    fig = plt.figure(figsize=(7, 9))
    fig.suptitle(f'Grad-CAM Analysis: {image_name}',
                 fontsize=10, fontweight='bold', y=0.99)

    gs = GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.28,
                  top=0.95, bottom=0.06, left=0.08, right=0.96)

    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])
    ax2  = fig.add_subplot(gs[2, :])

    display_size = (IMG_WIDTH * 4, IMG_HEIGHT * 4)

    # (a) Original image
    original_resized = original_img.resize(display_size, Image.Resampling.LANCZOS)
    ax00.imshow(original_resized)
    ax00.set_title('(a) Original Image', fontsize=9, fontweight='bold', pad=4)
    ax00.axis('off')

    # (b) Grad-CAM Heatmap
    heatmap_upscaled = np.array(Image.fromarray(
        (heatmap * 255).astype(np.uint8)
    ).resize(display_size, Image.Resampling.BILINEAR)) / 255.0

    im = ax01.imshow(heatmap_upscaled, cmap='jet', vmin=0, vmax=1)
    ax01.set_title('(b) Grad-CAM Heatmap', fontsize=9, fontweight='bold', pad=4)
    ax01.axis('off')
    cbar = plt.colorbar(im, ax=ax01, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label('Activation', fontsize=7)

    # (c) Grad-CAM Overlay
    overlay, _ = create_gradcam_visualization(original_img, heatmap, alpha=0.5)
    ax10.imshow(overlay)
    ax10.set_title('(c) Grad-CAM Overlay', fontsize=9, fontweight='bold', pad=4)
    ax10.axis('off')

    # (d) Raw heatmap at model resolution
    ax11.imshow(heatmap, cmap='jet', interpolation='nearest', vmin=0, vmax=1)
    ax11.set_title(f'(d) Raw Heatmap '
                   f'({heatmap.shape[0]}\u00d7{heatmap.shape[1]})',
                   fontsize=9, fontweight='bold', pad=4)
    ax11.axis('off')

    # (e) Classification probabilities — full-width bar chart
    exp_scores = np.exp(class_scores - np.max(class_scores))
    probabilities = exp_scores / np.sum(exp_scores) * 100

    colors = ['#2ecc71' if i == predicted_class else '#3498db'
              for i in range(NUM_CLASSES)]
    bars = ax2.barh(range(NUM_CLASSES), probabilities, color=colors, height=0.55)
    ax2.set_yticks(range(NUM_CLASSES))
    ax2.set_yticklabels(CLASS_NAMES, fontsize=8)
    ax2.set_xlabel('Probability (%)', fontsize=8)
    ax2.set_title(
        f'(e) Classification Probabilities  '
        f'\u2014  Predicted: {CLASS_NAMES[predicted_class]}  '
        f'(Confidence: {probabilities[predicted_class]:.1f}%)',
        fontsize=9, fontweight='bold', pad=4)
    ax2.set_xlim(0, 118)
    ax2.tick_params(axis='x', labelsize=7)
    ax2.spines[['top', 'right']].set_visible(False)

    for bar, prob in zip(bars, probabilities):
        ax2.text(bar.get_width() + 1.2, bar.get_y() + bar.get_height() / 2,
                 f'{prob:.1f}%', va='center', fontsize=7.5)

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    plt.rcParams.update({'font.family': plt.rcParamsDefault.get('font.family', 'sans-serif'),
                         'font.size': plt.rcParamsDefault.get('font.size', 10)})

    return output_path


def create_all_layers_visualization(original_img, model, input_tensor, predicted_class,
                                     class_scores, output_path, image_name):
    """
    Create a portrait all-layers Grad-CAM figure suitable for IEEE format papers.

    Layout (4 rows x 3 cols, bottom row spans full width):
      Row 0: (a) Original  |  (b) layer0 64x64  |  (c) layer1 32x32
      Row 1: (d) layer2 16x16  |  (e) layer3 8x8  |  (f) layer4 4x4
      Row 2: (g) layer5 4x4  |  (h) layer2 Overlay  |  (i) layer3 Overlay
      Row 3: (j) Classification Probabilities  (full width)

    Output: 7 in wide x 13 in tall @ 300 dpi — fits IEEE double-column figure width.
    """
    from matplotlib.gridspec import GridSpec

    layer_names = ['layer0', 'layer1', 'layer2', 'layer3', 'layer4', 'layer5']
    layer_resolutions = ['64\u00d764', '32\u00d732', '16\u00d716', '8\u00d78', '4\u00d74', '4\u00d74']
    layer_labels = [
        'Low-level\n(edges/texture)',
        'Low-level\n(edges/texture)',
        'Mid-level\n(patterns)',
        'Semantic\nfeatures',
        'High-level\n(abstract)',
        'High-level\n(abstract)',
    ]

    # Generate heatmaps for all layers
    heatmaps = []
    for layer_name in layer_names:
        model.set_target_layer(layer_name)
        model.gradients = None
        model.activations = None

        input_clone = input_tensor.clone().requires_grad_(True)
        output = model(input_clone)

        model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, predicted_class] = 1
        output.backward(gradient=one_hot)

        gradients = model.get_activations_gradient()
        activations = model.get_activations()

        if gradients is not None and activations is not None:
            weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * activations, dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = cam - cam.min()
            if cam.max() > 0:
                cam = cam / cam.max()
            heatmap = cam.squeeze().detach().cpu().numpy()
        else:
            heatmap = np.ones((8, 8)) * 0.5

        heatmaps.append(heatmap)

    # IEEE-friendly base font size
    plt.rcParams.update({'font.family': 'serif', 'font.size': 8})

    fig = plt.figure(figsize=(7, 13))
    fig.suptitle(f'Grad-CAM All Layers Analysis: {image_name}',
                 fontsize=10, fontweight='bold', y=0.995)

    gs = GridSpec(4, 3, figure=fig, hspace=0.38, wspace=0.25,
                  top=0.965, bottom=0.05, left=0.07, right=0.97)

    display_size = (IMG_WIDTH * 4, IMG_HEIGHT * 4)
    panel_labels = 'abcdefghij'
    title_fs = 8

    # ── Row 0: Original, layer0, layer1 ──────────────────────────────────────
    ax_orig = fig.add_subplot(gs[0, 0])
    original_resized = original_img.resize(display_size, Image.Resampling.LANCZOS)
    ax_orig.imshow(original_resized)
    ax_orig.set_title('(a) Input Image\n(64\u00d764 px)', fontsize=title_fs,
                      fontweight='bold', pad=3)
    ax_orig.axis('off')

    for col, (layer_idx, panel_label) in enumerate(zip([0, 1], ['b', 'c']), start=1):
        ax = fig.add_subplot(gs[0, col])
        hm_up = np.array(Image.fromarray(
            (heatmaps[layer_idx] * 255).astype(np.uint8)
        ).resize(display_size, Image.Resampling.BILINEAR)) / 255.0
        im = ax.imshow(hm_up, cmap='jet', vmin=0, vmax=1)
        ax.set_title(f'({panel_label}) {layer_names[layer_idx]}\n'
                     f'{layer_resolutions[layer_idx]}  {layer_labels[layer_idx]}',
                     fontsize=title_fs, fontweight='bold', pad=3)
        ax.axis('off')

    # ── Row 1: layer2, layer3, layer4 ────────────────────────────────────────
    for col, (layer_idx, panel_label) in enumerate(zip([2, 3, 4], ['d', 'e', 'f'])):
        ax = fig.add_subplot(gs[1, col])
        hm_up = np.array(Image.fromarray(
            (heatmaps[layer_idx] * 255).astype(np.uint8)
        ).resize(display_size, Image.Resampling.BILINEAR)) / 255.0
        im = ax.imshow(hm_up, cmap='jet', vmin=0, vmax=1)
        ax.set_title(f'({panel_label}) {layer_names[layer_idx]}\n'
                     f'{layer_resolutions[layer_idx]}  {layer_labels[layer_idx]}',
                     fontsize=title_fs, fontweight='bold', pad=3)
        ax.axis('off')

    # ── Row 2: layer5, layer2 overlay, layer3 overlay ────────────────────────
    # layer5 heatmap
    ax_l5 = fig.add_subplot(gs[2, 0])
    hm_up = np.array(Image.fromarray(
        (heatmaps[5] * 255).astype(np.uint8)
    ).resize(display_size, Image.Resampling.BILINEAR)) / 255.0
    im = ax_l5.imshow(hm_up, cmap='jet', vmin=0, vmax=1)
    ax_l5.set_title(f'(g) {layer_names[5]}\n'
                    f'{layer_resolutions[5]}  {layer_labels[5]}',
                    fontsize=title_fs, fontweight='bold', pad=3)
    ax_l5.axis('off')

    # Shared colorbar for all heatmap panels (anchored to layer5 panel)
    cbar_ax = fig.add_axes([0.355, 0.355, 0.008, 0.088])  # fine-tune if needed
    norm = plt.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap='jet', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Activation\nIntensity', fontsize=6.5)
    cbar.ax.tick_params(labelsize=6)

    for col, (layer_idx, panel_label) in enumerate(zip([2, 3], ['h', 'i']), start=1):
        ax = fig.add_subplot(gs[2, col])
        overlay, _ = create_gradcam_visualization(original_img, heatmaps[layer_idx], alpha=0.5)
        ax.imshow(overlay)
        ax.set_title(f'({panel_label}) {layer_names[layer_idx]} Overlay\n'
                     f'{layer_resolutions[layer_idx]}',
                     fontsize=title_fs, fontweight='bold', pad=3)
        ax.axis('off')

    # ── Row 3: Classification probabilities (full width) ─────────────────────
    ax_prob = fig.add_subplot(gs[3, :])
    exp_scores = np.exp(class_scores - np.max(class_scores))
    probabilities = exp_scores / np.sum(exp_scores) * 100

    colors = ['#2ecc71' if i == predicted_class else '#3498db'
              for i in range(NUM_CLASSES)]
    bars = ax_prob.barh(range(NUM_CLASSES), probabilities, color=colors, height=0.55)
    ax_prob.set_yticks(range(NUM_CLASSES))
    ax_prob.set_yticklabels(CLASS_NAMES, fontsize=8)
    ax_prob.set_xlabel('Probability (%)', fontsize=8)
    ax_prob.set_title(
        f'(j) Classification Probabilities  \u2014  '
        f'Predicted: {CLASS_NAMES[predicted_class]}  '
        f'(Confidence: {probabilities[predicted_class]:.1f}%)',
        fontsize=9, fontweight='bold', pad=4)
    ax_prob.set_xlim(0, 118)
    ax_prob.tick_params(axis='x', labelsize=7)
    ax_prob.spines[['top', 'right']].set_visible(False)

    for bar, prob in zip(bars, probabilities):
        ax_prob.text(bar.get_width() + 1.2, bar.get_y() + bar.get_height() / 2,
                     f'{prob:.1f}%', va='center', fontsize=7.5)

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    plt.rcParams.update({'font.family': plt.rcParamsDefault.get('font.family', 'sans-serif'),
                         'font.size': plt.rcParamsDefault.get('font.size', 10)})

    return output_path


def initialize_model_random():
    """Initialize model with random weights (for demo without checkpoint)"""
    model = HandwashCNN(num_classes=NUM_CLASSES)
    return model


def load_checkpoint(checkpoint_path, model):
    """
    Load weights from ai8x checkpoint file.
    
    The ai8x framework uses different layer naming conventions:
    - ai8x: conv1.op.weight, conv2.op.weight, ... (1-indexed)
    - our model: conv0.weight, conv1.weight, ... (0-indexed)
    
    This function maps the weights appropriately.
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Create mapping from ai8x names to our model names
        # ai8x uses 1-indexed conv layers (conv1, conv2, ..., conv6)
        # Our model uses 0-indexed (conv0, conv1, ..., conv5)
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
        
        # Build new state dict with mapped names
        new_state_dict = {}
        loaded_count = 0
        
        for ai8x_name, our_name in weight_mapping.items():
            if ai8x_name in state_dict:
                new_state_dict[our_name] = state_dict[ai8x_name]
                loaded_count += 1
                print(f"  Mapped: {ai8x_name} -> {our_name} {state_dict[ai8x_name].shape}")
        
        if loaded_count == 0:
            print("Warning: No weights could be mapped from checkpoint.")
            print("Available keys in checkpoint:")
            for k in list(state_dict.keys())[:10]:
                print(f"  {k}")
            return False
        
        # Load the mapped weights
        model.load_state_dict(new_state_dict, strict=False)
        print(f"\nLoaded {loaded_count} weight tensors from: {checkpoint_path}")
        
        # Set model to eval mode and disable batch norm tracking
        model.eval()
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
        
        return True
            
    except Exception as e:
        print(f"Warning: Could not load checkpoint: {e}")
        import traceback
        traceback.print_exc()
        print("Using random weights for demonstration.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM visualization for handwash classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s handwash_image.png
  %(prog)s handwash_image.jpg --output result_gradcam.png
  %(prog)s test.png --checkpoint trained/model.pth.tar --class 2

The output includes:
  - Original image
  - Grad-CAM heatmap showing model attention
  - Overlay visualization
  - Classification probabilities
        """
    )
    
    parser.add_argument(
        "input_image",
        help="Path to input image file"
    )
    
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output path for Grad-CAM visualization (default: <input>_gradcam.png)"
    )
    
    parser.add_argument(
        "--checkpoint", "-c",
        default=None,
        help="Path to PyTorch checkpoint file (.pth.tar)"
    )
    
    parser.add_argument(
        "--class", "-t",
        dest="target_class",
        type=int,
        default=None,
        help="Target class for Grad-CAM (default: predicted class)"
    )
    
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Overlay blend factor (default: 0.5)"
    )
    
    parser.add_argument(
        "--layer",
        type=str,
        default="layer3",
        choices=["layer0", "layer1", "layer2", "layer3", "layer4", "layer5"],
        help="Target layer for Grad-CAM visualization (default: layer3). "
             "Activations are captured AFTER ReLU. "
             "layer0=64x64, layer1=32x32, layer2=16x16, layer3=8x8, layer4/5=4x4."
    )
    
    parser.add_argument(
        "--all-layers",
        action="store_true",
        help="Generate visualization showing ALL layers (layer0-layer5) in one comparison image. "
             "This overrides --layer option and shows each layer's Grad-CAM side by side."
    )
    
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Output only the overlay image instead of full analysis figure"
    )
    
    args = parser.parse_args()
    
    # Check input file
    if not os.path.isfile(args.input_image):
        print(f"Error: Input file not found: {args.input_image}")
        sys.exit(1)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        base_name = os.path.splitext(args.input_image)[0]
        output_path = f"{base_name}_gradcam.png"
    
    print(f"Loading image: {args.input_image}")
    
    # Load and preprocess image
    input_tensor, original_img = load_and_preprocess_image(args.input_image)
    
    # Initialize model
    print("Initializing model...")
    model = initialize_model_random()
    
    # Load checkpoint if provided
    if args.checkpoint:
        load_checkpoint(args.checkpoint, model)
    else:
        print("Note: No checkpoint provided. Using random weights for demonstration.")
        print("      For accurate results, provide --checkpoint path/to/model.pth.tar")
    
    # Generate Grad-CAM
    print(f"Generating Grad-CAM visualization...")
    
    # Create visualization
    if args.all_layers:
        # Generate all-layers comparison visualization
        print("Generating ALL LAYERS comparison view...")
        
        # First do a forward pass to get prediction and scores
        model.set_target_layer('layer3')
        gradcam = GradCAM(model, target_layer='layer3')
        _, predicted_class, class_scores = gradcam.generate(
            input_tensor, 
            target_class=args.target_class
        )
        
        image_name = os.path.basename(args.input_image)
        create_all_layers_visualization(
            original_img, model, input_tensor, predicted_class, class_scores,
            output_path, image_name
        )
    elif args.simple:
        # Just the overlay image for single layer
        print(f"Using single layer: {args.layer}")
        gradcam = GradCAM(model, target_layer=args.layer)
        heatmap, predicted_class, class_scores = gradcam.generate(
            input_tensor, 
            target_class=args.target_class
        )
        overlay, _ = create_gradcam_visualization(original_img, heatmap, args.alpha)
        overlay.save(output_path)
    else:
        # Full analysis figure for single layer
        print(f"Using single layer: {args.layer}")
        gradcam = GradCAM(model, target_layer=args.layer)
        heatmap, predicted_class, class_scores = gradcam.generate(
            input_tensor, 
            target_class=args.target_class
        )
        image_name = os.path.basename(args.input_image)
        create_visualization_figure(
            original_img, heatmap, predicted_class, class_scores,
            output_path, image_name
        )
    
    # Get prediction info for display
    if not args.all_layers:
        pass  # Already have predicted_class and class_scores
    
    print(f"\n{'='*50}")
    print(f"Grad-CAM visualization saved: {output_path}")
    print(f"{'='*50}")
    print(f"Predicted Class: {predicted_class} ({CLASS_NAMES[predicted_class]})")
    
    # Show probabilities
    exp_scores = np.exp(class_scores - np.max(class_scores))
    probabilities = exp_scores / np.sum(exp_scores) * 100
    print(f"Confidence: {probabilities[predicted_class]:.1f}%")
    if args.all_layers:
        print(f"Visualization includes: layer0, layer1, layer2, layer3, layer4, layer5")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
