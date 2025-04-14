#!/usr/bin/env python3
"""
Animator2D-v1.0.0 Training Code
--------------------------------
This script is designed to train the Animator2D-v1.0.0 model for generating pixel art sprite animations.
It is compatible with Google Colab and handles the local dataset located in /content/dataset/images.
"""

# Google Colab Setup - Comment this section if running locally
# ===========================================================
import os
print("Checking environment and installing dependencies...")

# Install required packages for Google Colab
try:
    import google.colab
    IN_COLAB = True
    # Install dependencies
    !pip install torch torchvision transformers Pillow tqdm matplotlib tensorboard scikit-image imageio
    print("Google Colab detected, dependencies installed.")
    
    # Instructions for dataset loading
    print("""
    DATASET LOADING INSTRUCTIONS:
    -----------------------------
    1. Upload the dataset manually: Click on the folder icon on the left sidebar, 
       then upload the dataset to a folder named 'dataset/images'
    
    OR
    
    2. Mount Google Drive:
       from google.colab import drive
       drive.mount('/content/drive')
       
       Then copy your dataset to the correct location:
       !cp -r /content/drive/MyDrive/path/to/dataset/images /content/dataset/images
       
    Make sure your dataset structure follows the required format with spritesheet_X folders 
    containing frame_XX.png files and sprite_metadata.json in the dataset/images directory.
    """)
except ImportError:
    IN_COLAB = False
    print("Running in a local environment.")

# Common imports for both environments
# ==================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import save_image, make_grid

import numpy as np
import random
import json
import math
import time
import glob
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageChops
from matplotlib import pyplot as plt
from collections import defaultdict
import re
from transformers import BertTokenizer, BertModel, T5Tokenizer, T5EncoderModel
import imageio
from sklearn.model_selection import train_test_split

# Set seeds for reproducibility
# =============================
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed()

# Device configuration
# ===================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration
# ============
class Config:
    # Dataset
    DATA_DIR = '/content/dataset/images' if IN_COLAB else '/Users/lorenzo/Documents/GitHub/Animator2D/dataset/images'
    METADATA_FILE = os.path.join('/content/dataset/sprite_metadata.json')
    OUTPUT_DIR = '/content/output' if IN_COLAB else './output'
    
    # Model
    MODEL_NAME = "Animator2D-v1.0.0"
    IMAGE_SIZE = 64  # Base size for sprites
    MAX_FRAMES = 24  # Maximum frames to process
    BACKGROUND_COLOR = [128, 128, 128]  # Gray background to remove
    ALPHA_THRESHOLD = 10  # Threshold for alpha transparency
    
    # Training
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-4
    BETAS = (0.9, 0.999)
    WEIGHT_DECAY = 1e-5
    NUM_EPOCHS = 50
    VAL_SPLIT = 0.1
    NUM_WORKERS = 2
    
    # Model Architecture
    BERT_MODEL = 'bert-base-uncased'
    HIDDEN_DIM = 256
    TRANSFORMER_LAYERS = 4
    TRANSFORMER_HEADS = 8
    
    # Paths
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
    SAMPLE_DIR = os.path.join(OUTPUT_DIR, 'samples')
    LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')
    
    # Create necessary directories
    @staticmethod
    def create_dirs():
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(Config.SAMPLE_DIR, exist_ok=True)
        os.makedirs(Config.LOG_DIR, exist_ok=True)
        os.makedirs(Config.DATA_DIR, exist_ok=True)

# Create required directories
Config.create_dirs()

# Dataset Preparation
# ==================
class SpriteAnimationDataset(Dataset):
    def __init__(self, data_dir, metadata_file, transform=None, max_frames=24):
        """
        Dataset for sprite animations.
        
        Args:
            data_dir: Directory containing sprite folders
            metadata_file: Path to sprite_metadata.json
            transform: Image transformations to apply
            max_frames: Maximum number of frames to consider
        """
        self.data_dir = data_dir
        self.max_frames = max_frames
        self.transform = transform
        
        # Load metadata
        try:
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
            print(f"Loaded metadata from {metadata_file}")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading metadata: {e}")
            self.metadata = {}
        
        # Find all sprite folders
        self.sprite_folders = []
        pattern = os.path.join(data_dir, "spritesheet_*")
        for folder_path in glob.glob(pattern):
            folder_name = os.path.basename(folder_path)
            # Get the spritesheet number
            match = re.search(r"spritesheet_(\d+)", folder_name)
            if match:
                spritesheet_num = match.group(1)
                if spritesheet_num in self.metadata:
                    # Check if folder contains at least 2 frames
                    frame_files = sorted(glob.glob(os.path.join(folder_path, "frame_*.png")))
                    if len(frame_files) >= 2:
                        self.sprite_folders.append(folder_path)
        
        print(f"Found {len(self.sprite_folders)} valid sprite animation folders")
        
        # Group folders by sprite dimensions for efficient batching
        self.dimension_groups = defaultdict(list)
        self.sprite_dimensions = {}
        
        for folder in self.sprite_folders:
            # Check the first frame to get dimensions
            first_frame_path = os.path.join(folder, "frame_0.png")
            if os.path.exists(first_frame_path):
                try:
                    with Image.open(first_frame_path) as img:
                        width, height = img.size
                        dimension_key = (width, height)
                        self.sprite_dimensions[folder] = dimension_key
                        self.dimension_groups[dimension_key].append(folder)
                except Exception as e:
                    print(f"Error processing {first_frame_path}: {e}")
        
        # Create training entries based on dimension groups
        self.samples = []
        for dimension, folders in self.dimension_groups.items():
            for folder in folders:
                folder_name = os.path.basename(folder)
                match = re.search(r"spritesheet_(\d+)", folder_name)
                if match:
                    spritesheet_num = match.group(1)
                    if spritesheet_num in self.metadata:
                        frame_files = sorted(glob.glob(os.path.join(folder, "frame_*.png")))
                        if len(frame_files) >= 2:  # Need at least 2 frames (input + target)
                            self.samples.append({
                                'folder': folder,
                                'metadata': self.metadata[spritesheet_num],
                                'dimensions': dimension,
                                'num_frames': min(len(frame_files), self.max_frames)
                            })
        
        print(f"Created {len(self.samples)} training samples")
        
        # Report dimension statistics
        print("Sprite dimension groups:")
        for dim, folders in self.dimension_groups.items():
            print(f"  {dim}: {len(folders)} spritesheets")
    
    def __len__(self):
        return len(self.samples)
    
    def remove_background(self, img):
        """
        Remove the background from a sprite image by converting the background color to transparent
        """
        # Convert PIL image to NumPy array
        img_array = np.array(img)
        
        # Check image dimensions and channels
        if len(img_array.shape) == 2:  # Grayscale
            # Convert grayscale to RGB
            img_array = np.stack((img_array,)*3, axis=-1)
        
        is_rgba = img_array.shape[-1] == 4
        
        # Extract RGB channels for comparison
        rgb_array = img_array[..., :3] if is_rgba else img_array
        
        # Create an alpha mask where background pixels are transparent
        alpha = np.ones((rgb_array.shape[0], rgb_array.shape[1]), dtype=np.uint8) * 255
        
        # Find background pixels (pixels matching background color within threshold)
        # Using only RGB channels for comparison
        bg_color = Config.BACKGROUND_COLOR[:3]  # Use only RGB part of background color
        bg_mask = np.all(np.abs(rgb_array - bg_color) < Config.ALPHA_THRESHOLD, axis=2)
        alpha[bg_mask] = 0
        
        # Add or update alpha channel
        if is_rgba:
            img_array[..., 3] = alpha  # Update existing alpha channel
        else:
            img_array = np.dstack((img_array, alpha))  # Add new alpha channel
        
        # Return as PIL image
        return Image.fromarray(img_array)
    
    def is_empty_sprite(self, img):
        """Check if an image is just background (all pixels are the same)"""
        if img.mode == "RGBA":
            # For RGBA, check if all pixels are transparent
            alpha_data = np.array(img.split()[3])
            return np.all(alpha_data < 10)  # All transparent
        
        # For RGB, check if all pixels are the same color
        img_array = np.array(img)
        return np.all(img_array == img_array[0, 0])
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        folder = sample['folder']
        metadata = sample['metadata']
        num_frames = sample['num_frames']
        
        # Load frames
        frames = []
        for i in range(num_frames):
            frame_path = os.path.join(folder, f"frame_{i}.png")
            if os.path.exists(frame_path):
                try:
                    with Image.open(frame_path) as img:
                        # Handle different image modes
                        if img.mode not in ['RGB', 'RGBA']:
                            img = img.convert('RGB')
                        
                        # Remove background
                        img = self.remove_background(img)
                        
                        # Skip if the sprite is empty (all transparent)
                        if self.is_empty_sprite(img):
                            continue
                        
                        # Apply transform if provided
                        if self.transform:
                            img = self.transform(img)
                        
                        frames.append(img)
                except Exception as e:
                    print(f"Error loading frame {frame_path}: {str(e)}")
                    # Continue to next frame instead of failing entire sample
        
        # If no valid frames were loaded, return a default empty item
        if len(frames) == 0:
            # Create a blank transparent image
            blank = Image.new("RGBA", (Config.IMAGE_SIZE, Config.IMAGE_SIZE), (0, 0, 0, 0))
            if self.transform:
                blank = self.transform(blank)
            
            # Create empty metadata
            empty_metadata = {
                'action': '',
                'direction': '',
                'character': '',
                'full description': ''
            }
            
            return {
                'first_frame': blank,
                'frames': [blank],
                'metadata': empty_metadata,
                'num_frames': 1
            }
        
        # Extract metadata fields, using empty strings if keys don't exist
        action = metadata.get('action', '')
        direction = metadata.get('direction', '')
        character = metadata.get('character', '')
        full_description = metadata.get('full description', '')
        
        # Return structured data
        return {
            'first_frame': frames[0],  # First frame is the input image
            'frames': frames,          # All frames for the animation sequence
            'metadata': {
                'action': action,
                'direction': direction,
                'character': character,
                'full description': full_description
            },
            'num_frames': len(frames)
        }

# Custom collate function for batching
def collate_sprites_with_padding(batch):
    """
    Custom collate function that pads frames to handle variable numbers of frames
    and resizes images to a common size.
    """
    # Filter out empty samples
    batch = [item for item in batch if item['num_frames'] > 0]
    if not batch:
        return None
    
    # Get batch size and max number of frames
    batch_size = len(batch)
    max_frames = max(item['num_frames'] for item in batch)
    
    # Always use resize approach for safety, since sprites might be slightly different sizes
    # Use the config size as the target size
    target_size = (Config.IMAGE_SIZE, Config.IMAGE_SIZE)
    
    # Resize first frames
    resized_first_frames = []
    for item in batch:
        # Convert tensor to PIL for resizing
        tensor = item['first_frame']
        # Convert from [-1,1] to [0,1] range
        img = (tensor.clamp(-1, 1) + 1) / 2.0
        img = img.permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
        
        # Create PIL image
        if img.shape[2] == 4:  # RGBA
            pil_img = Image.fromarray((img * 255).astype(np.uint8), mode='RGBA')
        else:  # RGB
            pil_img = Image.fromarray((img[:, :, :3] * 255).astype(np.uint8), mode='RGB')
        
        # Resize
        pil_img = pil_img.resize(target_size, Image.NEAREST)
        
        # Convert back to tensor
        if pil_img.mode == 'RGBA':
            tensor = torch.from_numpy(np.array(pil_img)).float() / 255.0
            tensor = tensor.permute(2, 0, 1)  # HWC -> CHW
        else:
            tensor = torch.from_numpy(np.array(pil_img)).float() / 255.0
            tensor = tensor.permute(2, 0, 1)  # HWC -> CHW
            # Add alpha channel if missing
            if tensor.shape[0] == 3:
                alpha = torch.ones((1, *tensor.shape[1:]))
                tensor = torch.cat([tensor, alpha], dim=0)
        
        # Normalize back to [-1, 1]
        tensor = tensor * 2.0 - 1.0
        resized_first_frames.append(tensor)
    
    # Stack resized first frames
    first_frames = torch.stack(resized_first_frames)
    
    # Process and pad all frames
    padded_frames = []
    frame_masks = []
    
    for item in batch:
        frames = item['frames']
        num_frames = len(frames)
        
        # Resize and stack all frames
        resized_frames = []
        for frame in frames:
            # Convert tensor to PIL for resizing
            tensor = frame
            # Convert from [-1,1] to [0,1] range
            img = (tensor.clamp(-1, 1) + 1) / 2.0
            img = img.permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
            
            # Create PIL image
            if img.shape[2] == 4:  # RGBA
                pil_img = Image.fromarray((img * 255).astype(np.uint8), mode='RGBA')
            else:  # RGB
                pil_img = Image.fromarray((img[:, :, :3] * 255).astype(np.uint8), mode='RGB')
            
            # Resize
            pil_img = pil_img.resize(target_size, Image.NEAREST)
            
            # Convert back to tensor
            if pil_img.mode == 'RGBA':
                tensor = torch.from_numpy(np.array(pil_img)).float() / 255.0
                tensor = tensor.permute(2, 0, 1)  # HWC -> CHW
            else:
                tensor = torch.from_numpy(np.array(pil_img)).float() / 255.0
                tensor = tensor.permute(2, 0, 1)  # HWC -> CHW
                # Add alpha channel if missing
                if tensor.shape[0] == 3:
                    alpha = torch.ones((1, *tensor.shape[1:]))
                    tensor = torch.cat([tensor, alpha], dim=0)
            
            # Normalize back to [-1, 1]
            tensor = tensor * 2.0 - 1.0
            resized_frames.append(tensor)
        
        # Stack resized frames
        try:
            stacked_frames = torch.stack(resized_frames)
        except RuntimeError as e:
            # If stacking fails, we'll resize the tensors to be exactly the same size
            print(f"Warning: Encountered frame size mismatch - standardizing frames to exactly {target_size}...")
            standardized_frames = []
            for frame in resized_frames:
                # Force conversion through PIL again to ensure exact dimensions
                img = (frame.clamp(-1, 1) + 1) / 2.0
                img = img.permute(1, 2, 0).cpu().numpy()
                pil_img = Image.fromarray((img * 255).astype(np.uint8), mode='RGBA')
                pil_img = pil_img.resize(target_size, Image.NEAREST)
                
                tensor = torch.from_numpy(np.array(pil_img)).float() / 255.0
                tensor = tensor.permute(2, 0, 1)
                tensor = tensor * 2.0 - 1.0
                standardized_frames.append(tensor)
                
            stacked_frames = torch.stack(standardized_frames)
        
        # Create padding if needed
        if num_frames < max_frames:
            # Pad with zeros
            padding = torch.zeros(
                (max_frames - num_frames, *stacked_frames.shape[1:]),
                device=stacked_frames.device,
                dtype=stacked_frames.dtype
            )
            stacked_frames = torch.cat([stacked_frames, padding], dim=0)
        
        # Create mask
        mask = torch.zeros(max_frames, dtype=torch.bool)
        mask[:num_frames] = 1
        
        padded_frames.append(stacked_frames)
        frame_masks.append(mask)
    
    # Stack all padded frames and masks
    all_frames = torch.stack(padded_frames)
    all_masks = torch.stack(frame_masks)
    
    # Collect metadata
    actions = [item['metadata']['action'] for item in batch]
    directions = [item['metadata']['direction'] for item in batch]
    characters = [item['metadata']['character'] for item in batch]
    descriptions = [item['metadata']['full description'] for item in batch]
    num_frames = torch.tensor([item['num_frames'] for item in batch], dtype=torch.long)
    
    return {
        'first_frames': first_frames,         # Shape: [batch_size, C, H, W]
        'all_frames': all_frames,             # Shape: [batch_size, max_frames, C, H, W]
        'frame_masks': all_masks,             # Shape: [batch_size, max_frames]
        'actions': actions,                   # List of strings
        'directions': directions,             # List of strings
        'characters': characters,             # List of strings
        'descriptions': descriptions,         # List of strings
        'num_frames': num_frames              # Shape: [batch_size]
    }

# Model Architecture
# =================

# Residual Block for the generator
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.instance_norm1 = nn.InstanceNorm2d(in_channels)
        self.instance_norm2 = nn.InstanceNorm2d(in_channels)
        
    def forward(self, x):
        residual = x
        x = F.leaky_relu(self.instance_norm1(self.conv1(x)), 0.2)
        x = self.instance_norm2(self.conv2(x))
        return x + residual

# Self Attention Module
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Reshape for matrix multiplication
        proj_query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, height * width)
        
        # Compute attention weights
        attention = torch.bmm(proj_query, proj_key)
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to values
        proj_value = self.value(x).view(batch_size, -1, height * width)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        # Apply gamma (learnable parameter) to the attention output
        out = self.gamma * out + x
        
        return out

# Transformer-based Text Encoder - using BERT
class TextEncoder(nn.Module):
    def __init__(self, output_dim=Config.HIDDEN_DIM):
        super(TextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(Config.BERT_MODEL)
        self.linear = nn.Linear(768, output_dim)  # BERT hidden size is 768
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use the [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0]
        return self.linear(cls_output)

# Image Encoder
class ImageEncoder(nn.Module):
    def __init__(self, input_channels=4, output_dim=Config.HIDDEN_DIM):
        super(ImageEncoder, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # Initial convolution: [batch, 4, 64, 64] -> [batch, 64, 32, 32]
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(64),
            
            # Second layer: [batch, 64, 32, 32] -> [batch, 128, 16, 16]
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(128),
            
            # Third layer: [batch, 128, 16, 16] -> [batch, 256, 8, 8]
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(256),
            
            # Self-attention layer
            SelfAttention(256),
            
            # Residual blocks
            ResidualBlock(256),
            ResidualBlock(256)
        )
        
        # Global average pooling and final FC layer
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, output_dim)
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_pool(x).view(x.size(0), -1)
        return self.fc(x)

# Transformer-based Frame Generator
class TransformerFrameGenerator(nn.Module):
    def __init__(self, 
                 input_dim=Config.HIDDEN_DIM*2,  # Text + Image features
                 hidden_dim=Config.HIDDEN_DIM,
                 num_layers=Config.TRANSFORMER_LAYERS,
                 num_heads=Config.TRANSFORMER_HEADS,
                 max_seq_len=Config.MAX_FRAMES):
        super(TransformerFrameGenerator, self).__init__()
        
        # Position encoding
        self.max_seq_len = max_seq_len
        self.position_embedding = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
        
        # Project input features to hidden dimension
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, features, num_frames):
        batch_size = features.size(0)
        
        # Repeat features for each frame
        # features shape: [batch_size, feature_dim]
        # expanded shape: [batch_size, max_seq_len, feature_dim]
        expanded = features.unsqueeze(1).repeat(1, self.max_seq_len, 1)
        
        # Project to hidden dimension
        projected = self.input_proj(expanded)
        
        # Add positional encoding
        x = projected + self.position_embedding
        
        # Create padding mask based on num_frames
        mask = torch.arange(self.max_seq_len, device=num_frames.device)[None, :] >= num_frames[:, None]
        
        # Apply transformer
        output = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        return output

# Frame Decoder - transforms features into images
class FrameDecoder(nn.Module):
    def __init__(self, input_dim=Config.HIDDEN_DIM, output_channels=4):
        super(FrameDecoder, self).__init__()
        
        # Initial linear projection to 4x4 spatial dimension
        self.linear = nn.Linear(input_dim, 256 * 4 * 4)
        
        # Upsampling layers
        self.decoder = nn.Sequential(
            # [batch, 256, 4, 4] -> [batch, 256, 8, 8]
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(256),
            ResidualBlock(256),
            
            # [batch, 256, 8, 8] -> [batch, 128, 16, 16]
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(128),
            ResidualBlock(128),
            
            # [batch, 128, 16, 16] -> [batch, 64, 32, 32]
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(64),
            SelfAttention(64),
            ResidualBlock(64),
            
            # [batch, 64, 32, 32] -> [batch, output_channels, 64, 64]
            nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output range: [-1, 1]
        )
    
    def forward(self, x):
        # x shape: [batch, seq_len, input_dim]
        batch_size, seq_len, _ = x.shape
        
        # Reshape for processing each frame separately
        x = x.contiguous().view(batch_size * seq_len, -1)
        
        # Initial projection and reshape to spatial
        x = self.linear(x)
        x = x.view(-1, 256, 4, 4)
        
        # Decode to image
        x = self.decoder(x)
        
        # Reshape back to sequence form
        x = x.view(batch_size, seq_len, x.size(1), x.size(2), x.size(3))
        
        return x

# Complete Animator2D Model
class Animator2DModel(nn.Module):
    def __init__(self):
        super(Animator2DModel, self).__init__()
        
        # Text encoder
        self.text_encoder = TextEncoder()
        
        # Image encoder for the first frame
        self.image_encoder = ImageEncoder(input_channels=4)  # RGBA
        
        # Frame generator
        self.frame_generator = TransformerFrameGenerator()
        
        # Frame decoder
        self.frame_decoder = FrameDecoder(output_channels=4)  # RGBA
        
        print(f"Initialized Animator2D model with parameters:")
        print(f"- Image size: {Config.IMAGE_SIZE}x{Config.IMAGE_SIZE}")
        print(f"- Max frames: {Config.MAX_FRAMES}")
        print(f"- Hidden dim: {Config.HIDDEN_DIM}")
    
    def forward(self, first_frame, input_ids, attention_mask, num_frames):
        """
        Forward pass of the model
        
        Args:
            first_frame: First frame tensor [batch_size, C, H, W]
            input_ids: Token IDs from BERT tokenizer
            attention_mask: Attention mask for BERT
            num_frames: Number of frames to generate for each sample [batch_size]
            
        Returns:
            Generated frames [batch_size, num_frames, C, H, W]
        """
        # Encode text
        text_features = self.text_encoder(input_ids, attention_mask)
        
        # Encode first frame
        image_features = self.image_encoder(first_frame)
        
        # Concatenate text and image features
        combined_features = torch.cat([text_features, image_features], dim=1)
        
        # Generate frame features
        frame_features = self.frame_generator(combined_features, num_frames)
        
        # Decode frames
        generated_frames = self.frame_decoder(frame_features)
        
        return generated_frames

# Training Functions
# ================

def train_one_epoch(model, train_loader, optimizer, criterion, tokenizer, epoch, writer):
    model.train()
    total_loss = 0
    correct_frames = 0
    total_frames = 0
    batch_count = 0
    
    with tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch") as pbar:
        for batch in pbar:
            if batch is None:  # Skip empty batches
                continue
                
            optimizer.zero_grad()
            
            # Get batch data and move to device
            first_frames = batch['first_frames'].to(device)  # [batch_size, C, H, W]
            all_frames = batch['all_frames'].to(device)      # [batch_size, max_frames, C, H, W]
            frame_masks = batch['frame_masks'].to(device)    # [batch_size, max_frames]
            num_frames = batch['num_frames'].to(device)      # [batch_size]
            
            # Get text descriptions and tokenize
            descriptions = batch['descriptions']
            tokenized = tokenizer(descriptions, padding=True, truncation=True, 
                                 return_tensors='pt', max_length=512)
            input_ids = tokenized['input_ids'].to(device)
            attention_mask = tokenized['attention_mask'].to(device)
            
            # Forward pass
            outputs = model(first_frames, input_ids, attention_mask, num_frames)
            
            # Calculate loss only on real frames (not padding)
            loss = 0
            batch_size = outputs.size(0)
            
            for b in range(batch_size):
                # Get the target frames for this sample (excluding the first frame)
                target = all_frames[b, 1:num_frames[b]]
                
                # Get the generated frames for this sample (excluding the first frame)
                pred = outputs[b, :num_frames[b]-1]
                
                if target.size(0) > 0:  # Ensure we have targets
                    # Ensure target and prediction have the same dimensions
                    if pred.shape != target.shape:
                        # Resize prediction to match target size
                        if pred.shape[2:] != target.shape[2:]:
                            resized_pred = F.interpolate(
                                pred, 
                                size=target.shape[2:], 
                                mode='bilinear', 
                                align_corners=False
                            )
                            # MSE loss
                            sample_loss = criterion(resized_pred, target)
                        else:
                            # Channel mismatch but spatial dimensions match
                            # Pad or trim channels
                            if pred.shape[1] > target.shape[1]:
                                # Trim extra channels from prediction
                                pred = pred[:, :target.shape[1], :, :]
                            elif pred.shape[1] < target.shape[1]:
                                # Pad prediction with zeros
                                padding = torch.zeros(
                                    pred.shape[0], 
                                    target.shape[1] - pred.shape[1], 
                                    pred.shape[2], 
                                    pred.shape[3], 
                                    device=pred.device
                                )
                                pred = torch.cat([pred, padding], dim=1)
                            
                            sample_loss = criterion(pred, target)
                    else:
                        # MSE loss
                        sample_loss = criterion(pred, target)
                    
                    # Additional LPIPS or perceptual loss could be added here
                    
                    # Add to total loss
                    loss += sample_loss
                    
                    # For logging: count correctly predicted frames
                    # A frame is "correct" if its pixel error is below a threshold
                    # This is a simple metric and may not be ideal for pixel art
                    with torch.no_grad():
                        if pred.shape != target.shape:
                            if pred.shape[2:] != target.shape[2:]:
                                resized_pred = F.interpolate(
                                    pred, 
                                    size=target.shape[2:], 
                                    mode='bilinear', 
                                    align_corners=False
                                )
                                pixel_error = torch.mean(torch.abs(resized_pred - target), dim=[1, 2, 3])
                            else:
                                # Handle channel mismatch
                                if pred.shape[1] > target.shape[1]:
                                    pred_comp = pred[:, :target.shape[1], :, :]
                                elif pred.shape[1] < target.shape[1]:
                                    padding = torch.zeros(
                                        pred.shape[0], 
                                        target.shape[1] - pred.shape[1], 
                                        pred.shape[2], 
                                        pred.shape[3], 
                                        device=pred.device
                                    )
                                    pred_comp = torch.cat([pred, padding], dim=1)
                                else:
                                    pred_comp = pred
                                
                                pixel_error = torch.mean(torch.abs(pred_comp - target), dim=[1, 2, 3])
                        else:
                            pixel_error = torch.mean(torch.abs(pred - target), dim=[1, 2, 3])
                        
                        correct_frames += torch.sum(pixel_error < 0.1).item()
                        total_frames += target.size(0)
            
            # Normalize by batch size
            if batch_size > 0:  # Avoid division by zero
                loss = loss / batch_size
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=True)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            batch_count += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss / batch_count:.4f}",
                'acc': f"{correct_frames / max(1, total_frames):.2%}"
            })
    
    # Calculate average metrics
    avg_loss = total_loss / max(1, batch_count)
    accuracy = correct_frames / max(1, total_frames)
    
    # Log to TensorBoard
    writer.add_scalar('train/loss', avg_loss, epoch)
    writer.add_scalar('train/accuracy', accuracy, epoch)
    
    return avg_loss, accuracy

def validate(model, val_loader, criterion, tokenizer, epoch, writer):
    model.eval()
    total_loss = 0
    correct_frames = 0
    total_frames = 0
    batch_count = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", unit="batch"):
            if batch is None:  # Skip empty batches
                continue
                
            # Get batch data and move to device
            first_frames = batch['first_frames'].to(device)
            all_frames = batch['all_frames'].to(device)
            frame_masks = batch['frame_masks'].to(device)
            num_frames = batch['num_frames'].to(device)
            
            # Get text descriptions and tokenize
            descriptions = batch['descriptions']
            tokenized = tokenizer(descriptions, padding=True, truncation=True, 
                                 return_tensors='pt', max_length=512)
            input_ids = tokenized['input_ids'].to(device)
            attention_mask = tokenized['attention_mask'].to(device)
            
            # Forward pass
            outputs = model(first_frames, input_ids, attention_mask, num_frames)
            
            # Calculate metrics
            loss = 0
            batch_size = outputs.size(0)
            
            for b in range(batch_size):
                target = all_frames[b, 1:num_frames[b]]
                pred = outputs[b, :num_frames[b]-1]
                
                if target.size(0) > 0:
                    # Ensure target and prediction have the same dimensions
                    if pred.shape != target.shape:
                        # Resize prediction to match target size
                        if pred.shape[2:] != target.shape[2:]:
                            resized_pred = F.interpolate(
                                pred, 
                                size=target.shape[2:], 
                                mode='bilinear', 
                                align_corners=False
                            )
                            # MSE loss
                            sample_loss = criterion(resized_pred, target)
                        else:
                            # Handle channel mismatch
                            if pred.shape[1] > target.shape[1]:
                                pred = pred[:, :target.shape[1], :, :]
                            elif pred.shape[1] < target.shape[1]:
                                padding = torch.zeros(
                                    pred.shape[0], 
                                    target.shape[1] - pred.shape[1], 
                                    pred.shape[2], 
                                    pred.shape[3], 
                                    device=pred.device
                                )
                                pred = torch.cat([pred, padding], dim=1)
                            
                            sample_loss = criterion(pred, target)
                    else:
                        sample_loss = criterion(pred, target)
                    
                    loss += sample_loss
                    
                    # Calculate accuracy with dimension checking
                    if pred.shape != target.shape:
                        if pred.shape[2:] != target.shape[2:]:
                            resized_pred = F.interpolate(
                                pred, 
                                size=target.shape[2:], 
                                mode='bilinear', 
                                align_corners=False
                            )
                            pixel_error = torch.mean(torch.abs(resized_pred - target), dim=[1, 2, 3])
                        else:
                            # Handle channel mismatch
                            if pred.shape[1] > target.shape[1]:
                                pred_comp = pred[:, :target.shape[1], :, :]
                            elif pred.shape[1] < target.shape[1]:
                                padding = torch.zeros(
                                    pred.shape[0], 
                                    target.shape[1] - pred.shape[1], 
                                    pred.shape[2], 
                                    pred.shape[3], 
                                    device=pred.device
                                )
                                pred_comp = torch.cat([pred, padding], dim=1)
                            else:
                                pred_comp = pred
                            
                            pixel_error = torch.mean(torch.abs(pred_comp - target), dim=[1, 2, 3])
                    else:
                        pixel_error = torch.mean(torch.abs(pred - target), dim=[1, 2, 3])
                    
                    correct_frames += torch.sum(pixel_error < 0.1).item()
                    total_frames += target.size(0)
            
            # Normalize by batch size
            if batch_size > 0:
                loss = loss / batch_size
            else:
                loss = torch.tensor(0.0, device=device)
            
            # Update statistics
            total_loss += loss.item()
            batch_count += 1
    
    # Calculate average metrics
    avg_loss = total_loss / max(1, batch_count)
    accuracy = correct_frames / max(1, total_frames)
    
    # Log to TensorBoard
    writer.add_scalar('val/loss', avg_loss, epoch)
    writer.add_scalar('val/accuracy', accuracy, epoch)
    
    return avg_loss, accuracy

def generate_sample_animations(model, val_loader, tokenizer, epoch, sample_dir, num_samples=4):
    """Generate and save sample animations for visualization"""
    model.eval()
    
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    
    # Get a batch from the validation set
    for batch in val_loader:
        if batch is None:
            continue
        break
    
    # Process a limited number of samples
    with torch.no_grad():
        # Get batch data and move to device
        first_frames = batch['first_frames'][:num_samples].to(device)
        all_frames_gt = batch['all_frames'][:num_samples].to(device)
        num_frames = batch['num_frames'][:num_samples].to(device)
        
        # Get text descriptions and tokenize
        descriptions = batch['descriptions'][:num_samples]
        tokenized = tokenizer(descriptions, padding=True, truncation=True, 
                             return_tensors='pt', max_length=512)
        input_ids = tokenized['input_ids'].to(device)
        attention_mask = tokenized['attention_mask'].to(device)
        
        # Forward pass
        outputs = model(first_frames, input_ids, attention_mask, num_frames)
        
        # For each sample, create a GIF
        for i in range(num_samples):
            n_frames = num_frames[i].item()
            
            # Get ground truth frames
            gt_frames = all_frames_gt[i, :n_frames].cpu()
            
            # Get generated frames (including the input frame)
            gen_frames = torch.cat([
                first_frames[i].unsqueeze(0).cpu(),
                outputs[i, :n_frames-1].cpu()
            ], dim=0)
            
            # Create directory for this epoch if it doesn't exist
            epoch_dir = os.path.join(sample_dir, f"epoch_{epoch}")
            os.makedirs(epoch_dir, exist_ok=True)
            
            # Convert tensors to PIL images and create GIFs
            gt_pil_frames = []
            gen_pil_frames = []
            
            # Function to convert tensor to PIL image
            def tensor_to_pil(tensor):
                # Convert from [-1,1] to [0,1]
                img = (tensor.clamp(-1, 1) + 1) / 2.0
                
                # Convert to uint8 [0,255]
                img = (img * 255).byte().permute(1, 2, 0).numpy()
                
                # Create PIL image
                if img.shape[2] == 4:  # RGBA
                    return Image.fromarray(img, mode='RGBA')
                else:  # RGB
                    return Image.fromarray(img[:, :, :3], mode='RGB')
            
            # Convert ground truth frames
            for j in range(n_frames):
                gt_pil_frames.append(tensor_to_pil(gt_frames[j]))
            
            # Convert generated frames
            for j in range(gen_frames.size(0)):
                gen_pil_frames.append(tensor_to_pil(gen_frames[j]))
            
            # Save GIFs
            gt_path = os.path.join(epoch_dir, f"sample_{i}_ground_truth.gif")
            gen_path = os.path.join(epoch_dir, f"sample_{i}_generated.gif")
            
            # Save GIFs with 100ms duration per frame (10 FPS)
            try:
                imageio.mimsave(gt_path, gt_pil_frames, duration=100)
                imageio.mimsave(gen_path, gen_pil_frames, duration=100)
                print(f"Saved sample animations for sample {i} to {epoch_dir}")
            except Exception as e:
                print(f"Error saving GIFs: {e}")
            
            # Save description
            desc_path = os.path.join(epoch_dir, f"sample_{i}_description.txt")
            with open(desc_path, 'w') as f:
                f.write(descriptions[i])

# Post-Processing Function
def post_process_animation(frames, color_palette=16):
    """Apply post-processing to quantize the pixel colors"""
    processed_frames = []
    
    for frame in frames:
        # Convert from tensor to PIL image
        if isinstance(frame, torch.Tensor):
            # Convert from [-1,1] to [0,1]
            img = (frame.clamp(-1, 1) + 1) / 2.0
            
            # Convert to uint8 [0,255]
            img = (img * 255).byte().permute(1, 2, 0).numpy()
            
            # Create PIL image
            if img.shape[2] == 4:  # RGBA
                pil_img = Image.fromarray(img, mode='RGBA')
            else:  # RGB
                pil_img = Image.fromarray(img[:, :, :3], mode='RGB')
        else:
            pil_img = frame
        
        # Quantize colors
        if color_palette > 0:
            if pil_img.mode == 'RGBA':
                # Extract alpha channel
                rgb, alpha = pil_img.convert('RGB'), pil_img.split()[3]
                
                # Quantize RGB part
                quantized = rgb.quantize(colors=color_palette)
                
                # Convert back to RGB
                quantized = quantized.convert('RGB')
                
                # Add alpha back
                quantized.putalpha(alpha)
            else:
                quantized = pil_img.quantize(colors=color_palette).convert('RGB')
        else:
            quantized = pil_img
        
        processed_frames.append(quantized)
    
    return processed_frames

# Main Training Loop
# ================
def main():
    # Initialize transforms
    transform = transforms.Compose([
        # Convert to tensor
        transforms.ToTensor(),
        # Normalize to [-1, 1] range
        transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5, 0.5])
    ])
    
    # Create dataset
    full_dataset = SpriteAnimationDataset(
        data_dir=Config.DATA_DIR,
        metadata_file=Config.METADATA_FILE,
        transform=transform,
        max_frames=Config.MAX_FRAMES
    )
    
    # Split into train and validation sets
    train_size = int((1 - Config.VAL_SPLIT) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    print(f"Training on {train_size} samples, validating on {val_size} samples")
    
    # Create data loaders - group by dimension if possible
    # Determine whether to use our dimension-based batching
    use_dimension_batching = hasattr(full_dataset, 'dimension_groups') and len(full_dataset.dimension_groups) > 0
    
    if use_dimension_batching:
        print("Using dimension-based batch grouping for more efficient training")
        
        # Create a sampler that groups by dimension
        dimension_samplers = []
        dimension_batch_sizes = {}
        
        for dimension, folders in full_dataset.dimension_groups.items():
            # Find indices of samples with this dimension
            dimension_indices = []
            for i in range(len(full_dataset.samples)):
                if full_dataset.samples[i]['dimensions'] == dimension:
                    dimension_indices.append(i)
            
            # Skip if no samples found for this dimension
            if not dimension_indices:
                continue
                
            # Calculate the batch size for this dimension group
            # (smaller batches for larger images to conserve memory)
            width, height = dimension
            pixels = width * height
            batch_size_factor = min(1.0, (64*64) / pixels)  # Scale batch size based on resolution
            batch_size = max(1, int(Config.BATCH_SIZE * batch_size_factor))
            dimension_batch_sizes[dimension] = batch_size
            
            # Create subset and sampler for this dimension
            subset_indices = [idx for idx in range(len(train_dataset)) 
                             if train_dataset.dataset.samples[train_dataset.indices[idx]]['dimensions'] == dimension]
            
            if subset_indices:
                dimension_samplers.append(subset_indices)
        
        # Create DataLoaders for each dimension group
        train_loaders = []
        for indices in dimension_samplers:
            # Create a subset
            subset = Subset(train_dataset, indices)
            
            # Get the dimension of this subset
            sample_idx = subset.indices[0]
            dimension = train_dataset.dataset.samples[train_dataset.indices[sample_idx]]['dimensions']
            batch_size = dimension_batch_sizes.get(dimension, Config.BATCH_SIZE)
            
            # Create DataLoader
            train_loaders.append(
                DataLoader(
                    subset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=Config.NUM_WORKERS,
                    collate_fn=collate_sprites_with_padding,
                    pin_memory=True if torch.cuda.is_available() else False,
                    drop_last=False
                )
            )
        
        # If no dimension samplers were created, fall back to standard approach
        if not train_loaders:
            print("Warning: No dimension groups created, falling back to standard DataLoader")
            train_loader = DataLoader(
                train_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=True,
                num_workers=Config.NUM_WORKERS,
                collate_fn=collate_sprites_with_padding,
                pin_memory=True if torch.cuda.is_available() else False,
                drop_last=False
            )
        else:
            # Use a ConcatDataset approach by iterating through loaders
            print(f"Created {len(train_loaders)} dimension-specific DataLoaders")
            # No need to create a ConcatDataset, we'll handle this in the training loop
    else:
        # Use standard DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            collate_fn=collate_sprites_with_padding,
            pin_memory=True if torch.cuda.is_available() else False,
            drop_last=False
        )
    
    # Validation loader (always use standard approach, with common resize)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_sprites_with_padding,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Initialize the model
    model = Animator2DModel().to(device)
    
    # Initialize BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained(Config.BERT_MODEL)
    
    # Initialize optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        betas=Config.BETAS,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=Config.NUM_EPOCHS
    )
    
    # Loss function - MSE for now, but could be changed to other losses
    criterion = nn.MSELoss()
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(Config.LOG_DIR)
    
    # Training loop
    best_val_loss = float('inf')
    
    print(f"Starting training for {Config.NUM_EPOCHS} epochs...")
    for epoch in range(Config.NUM_EPOCHS):
        # Train for one epoch
        if use_dimension_batching and 'train_loaders' in locals() and train_loaders:
            # Train with multiple dimension-specific loaders
            total_train_loss = 0
            total_train_acc = 0
            total_batches = 0
            
            for loader_idx, dimension_loader in enumerate(train_loaders):
                print(f"Training on dimension group {loader_idx+1}/{len(train_loaders)}")
                dim_train_loss, dim_train_acc = train_one_epoch(
                    model, dimension_loader, optimizer, criterion, tokenizer, epoch, writer
                )
                # Weight by number of batches in this loader
                loader_weight = len(dimension_loader)
                total_train_loss += dim_train_loss * loader_weight
                total_train_acc += dim_train_acc * loader_weight
                total_batches += loader_weight
            
            # Calculate weighted average
            train_loss = total_train_loss / max(1, total_batches)
            train_acc = total_train_acc / max(1, total_batches)
        else:
            # Standard training with a single loader
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, tokenizer, epoch, writer
            )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, tokenizer, epoch, writer
        )
        
        # Generate sample animations
        if (epoch + 1) % 5 == 0 or epoch == 0:
            generate_sample_animations(
                model, val_loader, tokenizer, epoch, Config.SAMPLE_DIR
            )
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)
        
        # Save model checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc
        }
        
        # Save regular checkpoint
        torch.save(checkpoint, os.path.join(Config.CHECKPOINT_DIR, f"{Config.MODEL_NAME}_checkpoint_epoch_{epoch}.pth"))
        
        # Save best model
        if is_best:
            torch.save(checkpoint, os.path.join(Config.CHECKPOINT_DIR, f"{Config.MODEL_NAME}_best.pth"))
            print(f"Epoch {epoch+1}: New best model saved with validation loss: {val_loss:.4f}")
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2%}")
        print(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2%}")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
    }, os.path.join(Config.CHECKPOINT_DIR, f"{Config.MODEL_NAME}_final.pth"))
    
    print("Training completed!")
    
    # Close TensorBoard writer
    writer.close()

if __name__ == "__main__":
    main()