#!/usr/bin/env python3
"""
Analyze Dataset Dimensions Script
--------------------------------
Questo script analizza le dimensioni di tutti gli sprite nel dataset.
Lo scopo principale è identificare gruppi di sprite con dimensioni simili
per facilitare l'addestramento in batch, dato che il modello funziona meglio
con immagini di dimensioni omogenee. Genera un report dettagliato che classifica
gli sprite per dimensioni e identifica potenziali problemi per il batching.
"""

import os
import sys
import json
from pathlib import Path
import logging
from PIL import Image
from collections import defaultdict

# Configura logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dataset_dimensions.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Dataset-Analyzer")

def analyze_dataset(dataset_path):
    """
    Analizza il dataset e raggruppa le immagini per dimensione.
    
    Args:
        dataset_path: Percorso alla directory del dataset
    """
    dataset_path = Path(dataset_path)
    images_path = dataset_path / 'image_transparent'
    metadata_path = dataset_path / 'sprite_metadata.json'
    
    # Verifica l'esistenza dei percorsi
    if not images_path.exists():
        logger.error(f"La cartella {images_path} non esiste")
        return
    
    if not metadata_path.exists():
        logger.error(f"Il file metadata {metadata_path} non esiste")
        return
    
    # Carica metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Dizionario per tenere traccia delle dimensioni
    dim_to_sequences = defaultdict(list)
    dimension_counts = defaultdict(int)
    
    # Analizza ogni sequenza
    for seq_id, sprite_data in metadata.items():
        folder_name = sprite_data.get('folder_name', f"spritesheet_{seq_id}")
        sprite_folder = images_path / folder_name
        
        if not sprite_folder.exists():
            logger.warning(f"Folder {folder_name} not found, skipping")
            continue
        
        # Cerca i file frame_*.png nella cartella
        available_frames = list(sprite_folder.glob("frame_*.png"))
        if not available_frames:
            logger.warning(f"No frame images found in {folder_name}, skipping")
            continue
        
        # Ordina i frame
        available_frames.sort()
        
        # Ottieni dimensioni del primo frame
        try:
            first_frame = Image.open(available_frames[0])
            width, height = first_frame.size
            dim_key = (width, height)
            
            # Registra questa sequenza
            dim_to_sequences[dim_key].append({
                'id': seq_id,
                'folder_name': folder_name,
                'num_frames': len(available_frames),
                'action': sprite_data.get('action', 'unknown'),
                'character': sprite_data.get('character', 'unknown')
            })
            
            dimension_counts[dim_key] += 1
        except Exception as e:
            logger.error(f"Error opening first frame of {folder_name}: {e}")
            continue
    
    # Mostra risultati
    logger.info(f"Found {len(dimension_counts)} unique dimension groups:")
    
    # Ordina per frequenza (dal più comune al meno comune)
    sorted_dims = sorted(dimension_counts.items(), key=lambda x: x[1], reverse=True)
    
    for dim, count in sorted_dims:
        width, height = dim
        logger.info(f"Dimension {width}x{height}: {count} sequences")
        
        # Mostra dettagli sulle sequenze
        logger.info(f"Sequences with dimension {width}x{height}:")
        for seq in dim_to_sequences[dim][:5]:  # Mostra solo le prime 5 per brevità
            logger.info(f"  - ID: {seq['id']}, Folder: {seq['folder_name']}, Character: {seq['character']}, Action: {seq['action']}")
        
        if len(dim_to_sequences[dim]) > 5:
            logger.info(f"  ... and {len(dim_to_sequences[dim]) - 5} more")
    
    # Analisi dimensioni con meno sequenze
    logger.info("\nDimension groups with few sequences (potential problems for batching):")
    for dim, count in sorted_dims:
        if count < 3:  # Mostra solo gruppi con meno di 3 sequenze
            width, height = dim
            logger.info(f"Dimension {width}x{height} has only {count} sequences:")
            for seq in dim_to_sequences[dim]:
                logger.info(f"  - ID: {seq['id']}, Folder: {seq['folder_name']}, Character: {seq['character']}, Action: {seq['action']}")
    
    return dim_to_sequences

if __name__ == "__main__":
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "/Users/lorenzo/Documents/GitHub/Animator2D/dataset"
    logger.info(f"Analyzing dataset at {dataset_path}")
    analyze_dataset(dataset_path)
    logger.info("Analysis complete. Check the dataset_dimensions.log file for details.")
