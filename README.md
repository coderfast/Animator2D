_Last update: 6 March 2025_
# 🎨 Animator2D

*Animator2D* is an AI-driven project I’ve embarked on to generate pixel-art sprite animations from textual descriptions. My goal is to create a tool that transforms prompts like “a knight in red armor attacking with a sword, facing right” into animated sequences—be it GIFs, sprite sheets, or videos—ready for indie game developers. For a detailed history of its evolution, check out [Change Log](CHANGELOG.md).

Link to my Hugging Face account: [https://huggingface.co/Lod34](https://huggingface.co/Lod34)

## 🌟 Project Journey

This project began as a passion for blending artificial intelligence with my love for indie games. I envisioned a system where developers could input a character description, action, direction, and frame count to produce pixel-art sprites with an ideal height of 35 pixels (within a 25-50 pixel range). What started as a straightforward idea has turned into a challenging journey of experimentation, setbacks, and learning. Here’s a detailed account of where I’ve been and where I’m headed.

### The First Step: Animator2D-v1.0.0-alpha (Development: Feb 21, 2025 - Release: Feb 22, 2025)
I kicked off *Animator2D-v1.0.0-alpha* on February 21, 2025, fueled by excitement. I chose BERT as the text encoder to process prompts and paired it with a simple convolutional generator for 64x64 pixel sprites. By February 22, I had a basic Gradio interface up, spitting out simulated outputs—yellow circles on blue backgrounds—just to test the pipeline. But the real results were a mess: chaotic pixels instead of animated sprites. BERT wasn’t cutting it for visual coherence, and the generator was too basic. It didn’t work, but it taught me I needed a more specialized approach.

### Simplifying and Iterating: Animator2D-mini-v1.0.0-alpha (Development: Feb 26, 2025 - Release: Mar 1, 2025)
After that initial stumble, I decided to simplify. On February 26, 2025, I started *Animator2D-mini-v1.0.0-alpha*, switching to CLIP for its text-image synergy and using a lighter generator. I adopted the `pawkanarek/spraix_1024` dataset from Hugging Face, intrigued by its sprite descriptions. Released on March 1, I tested three variants:
- **10e**: 10 epochs, vague 64x64 pixel shapes.
- **100e**: 100 epochs, slight improvements but still impractical.
- **250e**: 250 epochs, some stability at up to 128x128 pixels, yet no coherence.

I tweaked batch sizes (8-16) and learning rates (1e-4 to 2e-4), watching loss curves drop without usable outputs. It didn’t work, but I learned to handle datasets and PyTorch workflows, building a stronger technical foundation.

### An Ambitious Rewrite: Animator2D-v2.0.0-alpha (Development: Mar 2, 2025 - Release: Mar 3, 2025)
By March 2, 2025, I rewrote everything for *Animator2D-v2.0.0-alpha*. I swapped CLIP for T5, hoping for better text understanding, and added a *Frame Interpolator* for multi-frame animations. The generator grew more complex, and I upgraded the Gradio interface. Released on March 3, I deployed it to Hugging Face Spaces, only to find a “yellow ball on blue background” output. After days of debugging, I realized I’d uploaded the wrong `.pth` file. Even fixed, it didn’t work—animations were incoherent. This taught me deployment diligence, but also hinted my single-model approach might be flawed.

### A Fix with Hope: Animator2D-v3.0.0-alpha (Development & Release: Mar 6, 2025)
On March 6, 2025, I tackled *Animator2D-v3.0.0-alpha* as a fix for v2.0.0-alpha. I kept T5 and the *Frame Interpolator*, but added *Residual Blocks* and *Self-Attention* to the generator for better detail. Training got a boost with AdamW and Cosine Annealing on an 80/20 split of `pawkanarek/spraix_1024`. The Gradio interface now featured FPS control and GIF output, and I fixed the Hugging Face import (`Lod34/Animator2D`). Pixels started looking sprite-like, but it still didn’t work—animations lacked coherence. Progress, yes; success, no.

### Reflections on Setbacks
Post-v3, I paused. I’d gained skills in PyTorch, dataset management, and deployment, but practical results eluded me. Was the dataset too limited? Was a single model too ambitious? This frustration led to a pivot.

### A New Direction: Animator2D-v1.0.0 (Development Started: Mar 6, 2025)
Since March 6, 2025, I’ve been working on *Animator2D-v1.0.0*, a fresh start inspired by Da Vinci Resolve’s modular workflow. It’s in ideation, not yet functional, but here’s the plan:
1. **Creation**: Users create or import a base sprite. I’m exploring pre-trained models (e.g., Stable Diffusion for pixel-art) or breaking sprites into parts (head, arms) for animation ease. Balancing usability and complexity is key.
2. **Animation**: Set parameters—action, direction, frames—using `pawkanarek/spraix_1024` or richer datasets. This splits animation logic from creation.
3. **Generation**: Output in GIF, sprite sheet, or video format, with potential previews.

This modular approach feels promising, tackling one challenge at a time.

## 🛠️ Technical Details
- **Dataset**: `pawkanarek/spraix_1024`, preprocessed with resizing and normalization. It’s a start, but may need expansion.
- **Architectures**: BERT (v1.0.0-alpha), CLIP (mini-v1.0.0-alpha), T5 (v2.0.0-alpha, v3.0.0-alpha), with generators evolving to include *Residual Blocks* and *Self-Attention*.
- **Training**: Batch sizes 8-16, learning rates 1e-4/2e-4, up to 250 epochs. MSE Loss so far, but I’m eyeing alternatives.
- **Interface**: Gradio, from basic in v1.0.0-alpha to advanced with FPS in v3.0.0-alpha, hosted on Hugging Face Spaces.
- **Tech Stack**: PyTorch, Transformers, Gradio, Diffusers, PIL, NumPy. GPU (CUDA) when available, CPU otherwise.

## ⚡ Challenges Faced
- **Inconsistent Outputs**: Sprites don’t match prompts; animations are chaotic.
- **Dataset Limits**: `pawkanarek/spraix_1024` lacks variety for complex animations.
- **Deployment Hiccups**: The v2.0.0-alpha “yellow ball” fiasco taught me file verification.
- **Complexity**: A single-model approach overwhelmed me, leading to the modular shift.

## 🚀 Next Steps
- Build a prototype for *Animator2D-v1.0.0*’s three-phase structure.
- Test pre-trained models for *Creation*.
- Seek or create richer datasets for animation.
- Explore diffusion pipelines for robust generation.
- Enhance frame-to-frame coherence.

## 💭 Personal Reflections
*Animator2D* has been a rollercoaster. The “yellow ball” moment stung, and inconsistent outputs tested my resolve, but each step taught me something—neural networks, debugging, perseverance. It doesn’t work yet, but it’s a personal triumph of growth. The code’s on GitHub, and *v3.0.0-alpha* is testable on Hugging Face—not perfect, but a milestone. I’m committed to cracking this, one frame at a time.
