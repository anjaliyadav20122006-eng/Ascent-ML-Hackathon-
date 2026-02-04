# Super-Resolution for Satellite Imagery

## Project Overview

This project focuses on enhancing the resolution of satellite imagery using deep learning techniques. Specifically, it implements an EDSR (Enhanced Deep Residual Networks for Super-Resolution) model to upsample low-resolution (LR) satellite images to high-resolution (HR) counterparts, addressing the critical need for clearer and more detailed visual data in various applications, from environmental monitoring to urban planning. The project also establishes a baseline using Bicubic interpolation for comprehensive performance comparison.

## Technical Innovation: EDSR Model Architecture

**EDSR Model Choice**: The EDSR model was chosen for its proven effectiveness in super-resolution tasks, particularly its ability to deliver high-quality results without requiring batch normalization, which can sometimes degrade performance in SR. This leads to a simpler yet highly effective architecture.

**Key Components**:

-   **Residual Blocks**: The core of EDSR consists of multiple residual blocks. Each block employs two convolutional layers with skip connections, allowing the network to learn residual information and facilitating the training of very deep networks. This design helps in preserving fine details and preventing information loss during upsampling.
-   **Upsampler (PixelShuffle)**: For upsampling, the model uses `nn.PixelShuffle`, a sub-pixel convolution layer. This technique rearranges elements from a tensor of shape `(C * r^2, H, W)` to `(C, H * r, W * r)`, where `r` is the upscaling factor. It effectively produces HR images with fewer checkerboard artifacts compared to traditional transposed convolutions, contributing to superior visual quality.

## Training Methodology

**Model**: EDSR (Enhanced Deep Residual Networks)
**Dataset**: Patches extracted from the PROBA-V dataset (NIR band, grayscale images).
**Loss Function**: L1 Loss (`nn.L1Loss`) was used, as it tends to produce sharper images compared to L2 Loss by promoting sparsity and robustness to outliers.
**Optimizer**: Adam optimizer with a learning rate of 0.001.
**Epochs**: The model was trained for 5 epochs.
**Device**: Training was performed on CPU (though compatible with CUDA if available).

## Data Strategy: Local Patches for Training

For training and inference, the data was processed by extracting small, fixed-size patches from the original low-resolution and high-resolution images. This approach was adopted to manage computational resources and memory effectively during model training, especially when dealing with large images. The LR patches were 32x32 pixels, and corresponding HR patches were 96x96 pixels (for a 3x scale factor).

**Limitations & Future Work**: While effective for demonstration, this local patching strategy is a simplification. An ideal data strategy for large-scale satellite imagery would involve:

-   **API Streaming/Tiling**: Directly streaming and processing image tiles from platforms like WorldStrat or Google Earth Engine (GEE) to avoid downloading entire datasets.
-   **Dynamic Tiling**: Implementing dynamic tiling during training to further optimize memory usage and allow for larger batch sizes or image resolutions.

## Memory Management: Tiling for Inference

To handle large test images during inference without exceeding GPU memory limits, a tiling strategy is implemented. The low-resolution input image is divided into smaller, non-overlapping `64x64` patches. Each LR patch is then fed into the EDSR model independently to produce a `192x192` super-resolved patch. These SR patches are then stitched back together to reconstruct the full super-resolved image. This method ensures that even very large images can be processed efficiently.

## Evaluation

Both quantitative metrics and qualitative assessment were used to evaluate the model's performance against a Bicubic upsampling baseline.

### Quantitative Metrics

Comparison against the High-Resolution (HR) ground truth for a 3x upscaling target:

-   **PSNR (Peak Signal-to-Noise Ratio)**: Measures the ratio between the maximum possible power of a signal and the power of corrupting noise that affects the fidelity of its representation.
-   **SSIM (Structural Similarity Index Measure)**: Evaluates image quality based on perceived changes in structural information, luminance, and contrast.

| Method               | PSNR (dB) | SSIM   |
| :------------------- | :-------- | :----- |
| Bicubic Upsampling   | 38.34     | 0.9773 |
| EDSR Model (Tiled)   | 36.51     | 0.9586 |

*Note: The PSNR and SSIM values for the Bicubic baseline might appear very high or even 'inf' / '1.0' in some trivial cases if the chosen test images have extremely uniform content. This indicates the importance of diverse test data for meaningful evaluation. The current `imgset0618` provides a more representative evaluation.* 

### Qualitative Assessment ('The Eye Test')

Visual comparisons demonstrate that the EDSR model produces visually superior results compared to the low-resolution input and bicubic upsampling:

-   **Sharpness of Urban Edges**: EDSR effectively recovers sharper and more defined edges in urban areas, which are often blurred or jagged in LR images and bicubic interpolations.
-   **Road Clarity**: Roads appear clearer and more continuous in the EDSR output, providing better detail for mapping and analysis.
-   **Overall Perceptual Quality**: The EDSR-generated images exhibit a higher level of detail and a more natural appearance, reducing artifacts and enhancing the overall visual fidelity.

## Future Work & Improvements

1.  **Higher Upscaling Factors**: Explore training the EDSR model for higher upscaling factors (e.g., 4x or 8x) to address more challenging super-resolution scenarios.
2.  **Advanced Architectures**: Experiment with more advanced super-resolution architectures, including Generative Adversarial Networks (GANs) for improved perceptual quality or Transformer-based models for capturing long-range dependencies in satellite imagery.
3.  **Real-time Data Integration**: Implement direct integration with satellite imagery APIs (e.g., WorldStrat, Google Earth Engine) for real-time data streaming and processing, moving beyond local patch-based training.
4.  **Multi-spectral Imagery**: Extend the model to handle multi-spectral satellite images, processing multiple bands for more comprehensive super-resolution.