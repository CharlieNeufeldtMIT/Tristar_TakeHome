# Tristar Take Home Task
## Overview

### DetailVAE is an advanced Variational Autoencoder (VAE) designed for high-fidelity image reconstruction and precise defect detection. Leveraging residual blocks, multi-scale latent spaces, and perceptual loss, DetailVAE effectively captures intricate details in images, making it suitable for applications such as quality inspection and anomaly detection.

## Features

#### Image Preprocessing: Automated resizing, normalization, and organization of training and testing datasets.
#### Advanced VAE Architecture: Incorporates residual connections, multi-scale latent spaces, and detail enhancement layers for superior reconstruction quality.
#### Perceptual Loss Integration: Utilizes VGG16-based perceptual loss to maintain high-level feature consistency between original and reconstructed images.
#### Comprehensive Training Pipeline: Includes data augmentation, model checkpointing, and metric tracking for efficient training and evaluation.
#### Defect Detection: Generates defect maps with metrics like IoU, Precision, Recall, and F1-Score for evaluating defect detection performance.
#### Testing and Evaluation: Provides scripts to evaluate model performance on various categories with detailed metrics and visualizations.
#### Image Classification: Classifies images as "good" or "defective" based on reconstruction error thresholds.

## Architecture

### DetailVAE consists of three main components:

#### Encoder: Processes input images through convolutional layers with residual blocks and captures both global and detail-specific latent representations.
#### Latent Space: Utilizes dual latent vectors (global and detail) to encode comprehensive image information.
#### Decoder: Reconstructs images from the combined latent vectors, incorporating skip connections and detail enhancement layers to preserve image fidelity.
#### Additionally, the architecture integrates a Perceptual Loss module based on VGG16 to ensure that reconstructed images retain high-level features of the originals.

## The training script will:
### - Initialize the DetailVAE model and optimizer.
### - Apply data augmentation during training.
### - Save model checkpoints in the models directory.
### - Save reconstructed images and defect maps in the reconstructions directory.
### - Track and plot training and validation metrics in the metrics directory.
## Monitoring Training:
### Model Checkpoints: Saved in models/best_model.pth and periodically every 50 epochs.
### Reconstructed Images: Saved in reconstructions/train, reconstructions/validation, and reconstructions/test.
### Metrics: Training curves are saved as metrics/training_curves.png.

## The testing script will:
### - Load the trained DetailVAE model.
### - Evaluate reconstruction quality on "good" images.
### - Detect defects in "defective_painting" and "scratches" categories.
### - Calculate metrics such as MSE, PSNR, SSIM, IoU, Precision, Recall, and F1-Score.
### - Save visual comparisons and a heatmap of metrics in the test_results directory.
## Review Test Results:
### Reconstructed Images: Compare original and reconstructed images in test_results/good and test_results/<defect_type>.
### Defect Maps: Visualize defect detection performance.
### Metrics Heatmap: View test_results/metrics/metrics_heatmap.png for an overview of performance across categories.

## Classification
### Automatically classifies images as "good" or "defective" using the trained DetailVAE model.

