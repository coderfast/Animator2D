#!/usr/bin/env python3
"""
Animator2D Local App
-------------------
A Gradio interface for the Animator2D-v1.0.0 model.
This application allows users to generate pixel art sprite animations locally.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import gradio as gr
from transformers import BertTokenizer
import imageio
from pathlib import Path
import time
import importlib.util

# Import model architecture from training-code.py using importlib
spec = importlib.util.spec_from_file_location("training_code", "training-code.py")
training_code = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training_code)

# Import needed classes and functions
Animator2DModel = training_code.Animator2DModel
Config = training_code.Config
SelfAttention = training_code.SelfAttention
ResidualBlock = training_code.ResidualBlock
TextEncoder = training_code.TextEncoder
ImageEncoder = training_code.ImageEncoder
TransformerFrameGenerator = training_code.TransformerFrameGenerator
FrameDecoder = training_code.FrameDecoder
post_process_animation = training_code.post_process_animation

# Constants
MODEL_PATH = os.path.join("model", "Animator2D-v1.0.0.pth")
OUTPUT_DIR = os.path.join("output_samples")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained(Config.BERT_MODEL)


def load_model(model_path=MODEL_PATH):
    """Load the trained Animator2D model"""
    print(f"Loading model from {model_path}...")
    model = Animator2DModel().to(DEVICE)
    
    try:
        checkpoint = torch.load(model_path, map_location=DEVICE)
        
        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded model checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
        else:
            model.load_state_dict(checkpoint)
            print("Loaded model weights")
            
        model.eval()
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None


def preprocess_image(image, target_size=(64, 64)):
    """Preprocess the input image for the model"""
    # Resize image to target size
    image = image.resize(target_size, Image.NEAREST)
    
    # Convert image to RGBA if it's not already
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    
    # Convert to numpy array and normalize to [-1, 1]
    img_array = np.array(image).astype(np.float32) / 127.5 - 1.0
    
    # Convert to tensor and add batch dimension
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    
    return img_tensor


def remove_background(img, bg_color=[128, 128, 128], alpha_threshold=10):
    """Remove the background from a sprite image"""
    # Convert PIL image to NumPy array
    img_array = np.array(img)
    
    # Check image dimensions and channels
    if len(img_array.shape) == 2:  # Grayscale
        img_array = np.stack((img_array,)*3, axis=-1)
    
    is_rgba = img_array.shape[-1] == 4
    
    # Extract RGB channels for comparison
    rgb_array = img_array[..., :3] if is_rgba else img_array
    
    # Create an alpha mask where background pixels are transparent
    alpha = np.ones((rgb_array.shape[0], rgb_array.shape[1]), dtype=np.uint8) * 255
    
    # Find background pixels (pixels matching background color within threshold)
    bg_mask = np.all(np.abs(rgb_array - bg_color) < alpha_threshold, axis=2)
    alpha[bg_mask] = 0
    
    # Add or update alpha channel
    if is_rgba:
        img_array[..., 3] = alpha  # Update existing alpha channel
    else:
        img_array = np.dstack((img_array, alpha))  # Add new alpha channel
    
    # Return as PIL image
    return Image.fromarray(img_array)


def generate_animation(
    model, 
    first_frame, 
    prompt, 
    action="", 
    direction="", 
    character="", 
    num_frames=8,
    color_palette=16
):
    """
    Generate an animation from the first frame and text description
    
    Args:
        model: The Animator2D model
        first_frame: PIL image of the first frame
        prompt: Text description of the animation
        action: Action type (e.g., "walk", "attack")
        direction: Direction (e.g., "left", "right")
        character: Character type (e.g., "warrior", "mage")
        num_frames: Number of frames to generate
        color_palette: Number of colors for post-processing (0 = no quantization)
    
    Returns:
        List of PIL images representing the animation frames
    """
    # Preprocess the image
    first_frame = remove_background(first_frame)
    img_tensor = preprocess_image(first_frame).to(DEVICE)
    
    # Create full description by combining all text inputs
    full_description = prompt
    if action:
        full_description += f" Action: {action}."
    if direction:
        full_description += f" Direction: {direction}."
    if character:
        full_description += f" Character: {character}."
    
    # Tokenize the text
    tokenized = tokenizer(
        full_description, 
        padding=True, 
        truncation=True, 
        return_tensors='pt', 
        max_length=512
    )
    input_ids = tokenized['input_ids'].to(DEVICE)
    attention_mask = tokenized['attention_mask'].to(DEVICE)
    
    # Limit number of frames
    num_frames = min(max(2, num_frames), Config.MAX_FRAMES)
    
    # Create tensor for number of frames
    num_frames_tensor = torch.tensor([num_frames], dtype=torch.long, device=DEVICE)
    
    # Generate frames
    with torch.no_grad():
        outputs = model(img_tensor, input_ids, attention_mask, num_frames_tensor)
    
    # Process outputs
    generated_frames = []
    
    # Add the first frame (input)
    first_frame_tensor = img_tensor.squeeze(0)
    generated_frames.append(first_frame_tensor.cpu())
    
    # Add generated frames
    for i in range(min(num_frames - 1, outputs.size(1))):
        frame = outputs[0, i]
        generated_frames.append(frame.cpu())
    
    # Convert tensors to PIL images
    pil_frames = []
    for frame_tensor in generated_frames:
        # Convert from [-1, 1] to [0, 1]
        img = (frame_tensor.clamp(-1, 1) + 1) / 2.0
        
        # Convert to uint8 [0, 255]
        img = (img * 255).byte().permute(1, 2, 0).numpy()
        
        # Create PIL image
        if img.shape[2] == 4:  # RGBA
            pil_img = Image.fromarray(img, mode='RGBA')
        else:  # RGB
            pil_img = Image.fromarray(img[:, :, :3], mode='RGB')
        
        pil_frames.append(pil_img)
    
    # Apply post-processing
    if color_palette > 0:
        pil_frames = post_process_animation(pil_frames, color_palette)
    
    return pil_frames


def save_animation(frames, fps=10, output_path=None):
    """Save frames as a GIF animation"""
    if output_path is None:
        # Generate a unique filename based on timestamp
        timestamp = int(time.time())
        output_path = os.path.join(OUTPUT_DIR, f"animation_{timestamp}.gif")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as GIF
    imageio.mimsave(output_path, frames, duration=1000/fps)
    
    return output_path


def generate_and_save(
    image, prompt, action, direction, character, num_frames, fps, color_palette
):
    """Generate animation and save as GIF (for Gradio interface)"""
    global model
    
    # Check if model is loaded
    if model is None:
        return None, "Model not loaded. Please check the model path and try again."
    
    try:
        # Generate animation frames
        frames = generate_animation(
            model,
            image,
            prompt,
            action,
            direction,
            character,
            num_frames,
            color_palette
        )
        
        # Save as GIF
        output_path = save_animation(frames, fps)
        
        # Return the GIF path and success message
        return output_path, f"Animation generated successfully! Saved to {output_path}"
    except Exception as e:
        return None, f"Error generating animation: {str(e)}"


# Load the model once at startup
model = load_model()

# Create the Gradio interface
with gr.Blocks(title="Animator2D Local App") as app:
    gr.Markdown(
        """
        # 🎨 Animator2D Local Application
        
        Upload a sprite image and generate pixel art animations!
        
        ## How to use:
        1. Upload your sprite image (first frame)
        2. Enter a description of the animation
        3. Set optional parameters (action, direction, etc.)
        4. Click "Generate Animation"
        
        *Note: The model works best with pixel art sprites with dimensions around 64x64 pixels.*
        """
    )
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                type="pil", 
                label="Upload Sprite (First Frame)",
                image_mode="RGBA"
            )
            
            prompt = gr.Textbox(
                label="Animation Description", 
                placeholder="Describe the animation (e.g., 'A knight swinging a sword')"
            )
            
            with gr.Row():
                action = gr.Textbox(
                    label="Action", 
                    placeholder="walk, run, attack, idle, etc."
                )
                direction = gr.Textbox(
                    label="Direction", 
                    placeholder="left, right, up, down, etc."
                )
            
            character = gr.Textbox(
                label="Character Type", 
                placeholder="warrior, mage, monster, etc."
            )
            
            with gr.Row():
                num_frames = gr.Slider(
                    minimum=2,
                    maximum=24,
                    step=1,
                    value=8,
                    label="Number of Frames"
                )
                fps = gr.Slider(
                    minimum=1,
                    maximum=30,
                    step=1,
                    value=10,
                    label="Frames Per Second"
                )
            
            color_palette = gr.Slider(
                minimum=0,
                maximum=64,
                step=1,
                value=16,
                label="Color Palette Size (0 = no quantization)"
            )
            
            generate_btn = gr.Button("Generate Animation", variant="primary")
        
        with gr.Column():
            output_animation = gr.Image(
                type="filepath", 
                label="Generated Animation"
            )
            output_message = gr.Textbox(label="Status")
    
    # Set up the button click event
    generate_btn.click(
        generate_and_save,
        inputs=[
            input_image, prompt, action, direction, character, 
            num_frames, fps, color_palette
        ],
        outputs=[output_animation, output_message]
    )
    
    gr.Markdown(
        """
        ## Tips for Best Results:
        - Use pixel art sprites with transparent backgrounds
        - Keep sprites around 64x64 pixels in size
        - Be specific in your descriptions
        - Try different color palette values for best results
        - If results look weird, try adjusting the number of frames
        
        ## Credits:
        Animator2D-v1.0.0 by Lorenzo
        """
    )

# Launch the app
if __name__ == "__main__":
    app.launch()