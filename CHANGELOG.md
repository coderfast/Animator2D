# Changelog - Animator2D

All notable changes to the *Animator2D* project are documented in this file. This project aims to generate pixel-art sprite animations from textual descriptions, evolving through multiple iterations to support indie game developers.

## [Animator2D-v1.1] - In Development
- **Development Started**: April 6, 2025 (post-v1 completion)
- **Release**: Not yet released
- **New Direction**:
  - Online version hosted on Hugging Face.
  - Utilizes Hugging Face for dataset access and deploys the Gradio interface via a Hugging Face Space.
  - Uses a custom dataset (`Loacky/sprite-animation`, [link](https://huggingface.co/datasets/Loacky/sprite-animation)), derived from `pawkanarek/spraix_1024` by addressing its issues:
    - Original sprite sheets were single images, requiring manual cropping into individual frames organized in folders.
    - Sprite pixels didn’t match real image pixels (1 “fake” pixel = 3x3 real pixels), now corrected for true pixel-art fidelity.
  - Goal: Make the tool accessible online without local setup.
- **Status**: In progress, focusing on integration with Hugging Face infrastructure.

## [Animator2D-v1] - In Development
- **Development Started**: March 6, 2025
- **Release**: Not yet released
- **New Features**:
  - Local version running entirely on the user’s machine.
  - Training completed using a custom dataset (`Loacky/sprite-animation`, [link](https://huggingface.co/datasets/Loacky/sprite-animation)), created by refining `pawkanarek/spraix_1024`:
    - Converted sprite sheets from single images into individual frames, organized in folders.
    - Fixed pixel mismatch (1 “fake” pixel = 3x3 real pixels) for accurate pixel-art scale.
  - Modular approach in three phases:
    1. **Creation**: Generates a base sprite locally (e.g., via Stable Diffusion or first frame extraction), with a variable height parameter (e.g., 16px, 32px, 64px).
    2. **Animation**: Animates the base sprite using parameters (description, action, direction, frame count), trained on frame sequences.
    3. **Generation**: Outputs animations as GIF, sprite sheet, or video.
  - Gradio interface hosted locally, currently under development.
- **Technical Details**:
  - Training: Batch sizes 8-16, learning rate 1e-4/2e-4, epochs TBD (e.g., 50), MSE Loss, PyTorch with GPU (CUDA) or CPU.
  - Dataset frames mapped to base sprites and animation sequences.
- **Status**: Training completed; interface in progress, not fully functional.

## [Animator2D-v3-alpha] - 2025-03-06
- **Development Started**: March 6, 2025
- **Release**: March 6, 2025
- **Main Updates**:
  - Fixed *v2-alpha* with partial code rewrite.
  - Added Residual Blocks and Self-Attention to the generator for better detail and coherence.
  - Optimized Frame Interpolator for multi-frame animations (up to 16 frames, 256x256 pixels).
  - Used T5 as text encoder, trained with AdamW and Cosine Annealing on `pawkanarek/spraix_1024` (80/20 split).
  - Advanced Gradio interface with FPS control and GIF output.
- **Issues**:
  - Animations remained incoherent despite sprite-like pixels.
- **Status**: Not functional, deployed on Hugging Face Spaces (`Loacky/Animator2D`).

## [Animator2D-v2-alpha] - 2025-03-03
- **Development Started**: March 2, 2025
- **Release**: March 3, 2025
- **Main Updates**:
  - Complete rewrite from v1-alpha.
  - Introduced T5 text encoder and Frame Interpolator for multi-frame animations.
  - More complex generator architecture.
  - Enhanced Gradio interface.
- **Issues**:
  - Initial deployment on Hugging Face Spaces failed due to wrong `.pth` file upload, producing “yellow ball on blue” output. Fixed, but animations still incoherent.
- **Status**: Not functional.

## [Animator2D-mini-v1-alpha] - 2025-03-01
- **Development Started**: February 26, 2025
- **Release**: March 1, 2025
- **Main Updates**:
  - Simplified variant of v1-alpha for rapid testing.
  - Used CLIP as text encoder and a lightweight generator.
  - Variants:
    - *10e*: 10 epochs, batch size 8, learning rate 1e-4, 64x64 output, vague results.
    - *100e*: 100 epochs, visible improvement, still unusable.
    - *250e*: 250 epochs, batch size 16, learning rate 2e-4, up to 128x128, partial stability.
  - Trained on `pawkanarek/spraix_1024`.
- **Issues**:
  - Outputs lacked coherence for practical use.
- **Status**: Not functional.

## [Animator2D-v1-alpha] - 2025-02-22
- **Development Started**: February 21, 2025
- **Release**: February 22, 2025
- **Main Updates**:
  - First experimental version.
  - Used BERT as text encoder and a simple convolutional generator for 64x64 sprites.
  - Basic Gradio interface with simulated outputs (e.g., yellow circles on blue).
- **Issues**:
  - Produced incoherent pixel noise instead of sprites.
- **Status**: Not functional.

---

**Notes**:  
- Early versions (v1-alpha to v3-alpha) were experimental and non-functional, using `pawkanarek/spraix_1024` sprite sheets as-is, with limitations in frame organization and pixel scaling.  
- v1 and v1.1 introduce a custom dataset (`Loacky/sprite-animation`) to address these issues, splitting v1 for local use (training done, interface in progress) and v1.1 for online deployment (in development).
