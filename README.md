#### DetailVAE Model
The `vae_model.py` script defines the **Detail-Enhanced Variational Autoencoder (DetailVAE)**, a neural network designed for high-quality image reconstruction and defect detection. The model incorporates several advanced features to improve performance:

1. **Multi-Scale Latent Spaces**: Two latent spaces—one for capturing global features and another for fine-grained details—enable the model to retain intricate image characteristics.
2. **Residual Blocks with Skip Connections**: Enhances the model's ability to capture contextual information while minimizing reconstruction errors.
3. **Encoder-Decoder Architecture**: Combines downsampling layers with upsampling layers, paired with batch normalization, for efficient feature extraction and reconstruction.
4. **Perceptual Loss**: Uses a pre-trained VGG16 network to compute feature-level differences between original and reconstructed images.

#### VAE Trainer
The `train_with_validation.py` script provides a pipeline for training the DetailVAE. It includes data augmentation for improved generalization, a comprehensive training loop with checkpointing, and validation steps to evaluate the model's performance. Intermediate results, such as reconstructions and defect maps, are saved for qualitative assessment.

#### VAE Tester
The `test_vae.py` script evaluates the trained model on unseen datasets. It calculates metrics like Mean Squared Error (MSE), Peak Signal-to-Noise Ratio (PSNR), and F1-Score to measure reconstruction quality and defect detection accuracy. It also generates a heatmap summarizing the performance across multiple categories.

#### Image Classifier
The `classify_images.py` script uses a trained DetailVAE model to classify images into "good" or "defective" categories based on reconstruction error.

#### Validation Metrics and Visualizations
During the validation phase, several steps are performed to generate visualizations, such as `defective_painting_epoch_14.png`:

1. **Reconstruction and Defect Maps**:
   - Defective images are passed through the model to generate reconstructions.
   - A defect map is computed as the absolute pixel-wise difference between the original and reconstructed images, highlighting areas where the reconstruction deviates significantly.

2. **Refinement**:
   - Morphological operations, such as closing, are applied to the defect map to smooth and enhance defect regions.

3. **Visualization**:
   - The visualization combines the following elements for a clear comparison:
     - Original defective images.
     - Model-generated reconstructions.
     - Refined defect maps.
   - These are saved as grid images (e.g., `defective_painting_epoch_14.png`) in the `reconstructions/validation/` directory, allowing easy inspection of the model's performance for specific epochs.
  
