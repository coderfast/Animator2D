#!/usr/bin/env python3
"""
Remove Background Script
------------------------
Questo script rimuove lo sfondo grigio da tutti i frame degli spritesheet nella cartella dataset.
Il colore di sfondo è grigio ([128, 128, 128]) e viene rimosso rendendo quei pixel trasparenti.
Le immagini originali vengono mantenute e le versioni con sfondo trasparente sono salvate in
una nuova cartella 'image_transparent' dentro la directory dataset.
"""

import os
import glob
import json
import re
import numpy as np
from PIL import Image
from pathlib import Path
import shutil

# Configurazione
BACKGROUND_COLOR = [128, 128, 128]  # Colore di sfondo grigio
ALPHA_THRESHOLD = 10  # Soglia per la differenza di colore (tolleranza)

def remove_background(img):
    """
    Rimuove lo sfondo da un'immagine sprite convertendo il colore di sfondo in trasparenza.
    
    Args:
        img (PIL.Image): L'immagine di input
    
    Returns:
        PIL.Image: L'immagine con sfondo trasparente
    """
    # Converte immagine PIL in array NumPy
    img_array = np.array(img)
    
    # Controlla dimensioni e canali dell'immagine
    if len(img_array.shape) == 2:  # Scala di grigi
        # Converte scala di grigi a RGB
        img_array = np.stack((img_array,)*3, axis=-1)
    
    # Controlla se l'immagine ha già un canale alpha
    is_rgba = img_array.shape[-1] == 4
    
    # Estrai i canali RGB per il confronto
    rgb_array = img_array[..., :3] if is_rgba else img_array
    
    # Crea una maschera alpha dove i pixel di sfondo sono trasparenti
    alpha = np.ones((rgb_array.shape[0], rgb_array.shape[1]), dtype=np.uint8) * 255
    
    # Trova i pixel di sfondo (pixel che corrispondono al colore di sfondo entro la soglia)
    # Usa solo i canali RGB per il confronto
    bg_color = BACKGROUND_COLOR[:3]  # Usa solo la parte RGB del colore di sfondo
    bg_mask = np.all(np.abs(rgb_array - bg_color) < ALPHA_THRESHOLD, axis=2)
    alpha[bg_mask] = 0
    
    # Aggiungi o aggiorna il canale alpha
    if is_rgba:
        img_array[..., 3] = alpha  # Aggiorna il canale alpha esistente
    else:
        img_array = np.dstack((img_array, alpha))  # Aggiungi nuovo canale alpha
    
    # Restituisci come immagine PIL
    return Image.fromarray(img_array)

def process_spritesheet_frames(spritesheet_path, output_dir, verbose=True, save=True):
    """
    Processa tutti i frame in una cartella spritesheet per rimuovere lo sfondo.
    
    Args:
        spritesheet_path (str): Percorso alla cartella dello spritesheet
        output_dir (str): Directory di output dove salvare i frame processati
        verbose (bool): Se True, stampa messaggi dettagliati
        save (bool): Se True, salva i frame con sfondo trasparente
    
    Returns:
        int: Numero di frame processati
    """
    # Trova tutti i frame
    frame_pattern = os.path.join(spritesheet_path, "frame_*.png")
    frame_files = sorted(glob.glob(frame_pattern))
    
    if not frame_files:
        if verbose:
            print(f"Nessun frame trovato in {spritesheet_path}")
        return 0
    
    folder_name = os.path.basename(spritesheet_path)
    if verbose:
        print(f"Processando {folder_name} ({len(frame_files)} frame)...")
    
    # Crea directory di output corrispondente allo spritesheet
    output_spritesheet_dir = os.path.join(output_dir, folder_name)
    if save and not os.path.exists(output_spritesheet_dir):
        os.makedirs(output_spritesheet_dir, exist_ok=True)
    
    # Processa ogni frame
    processed_count = 0
    for frame_path in frame_files:
        try:
            # Carica l'immagine
            with Image.open(frame_path) as img:
                # Converti a RGB/RGBA se necessario
                if img.mode not in ['RGB', 'RGBA']:
                    img = img.convert('RGB')
                
                # Rimuovi sfondo
                processed_img = remove_background(img)
                
                # Salva l'immagine processata nella directory di output
                if save:
                    frame_name = os.path.basename(frame_path)
                    output_path = os.path.join(output_spritesheet_dir, frame_name)
                    processed_img.save(output_path)
                
                processed_count += 1
                
        except Exception as e:
            print(f"Errore nel processare {frame_path}: {e}")
            continue
    
    return processed_count

def process_all_spritesheets(verbose=True, save=True):
    """
    Processa tutti gli spritesheet nella directory delle immagini.
    
    Args:
        verbose (bool): Se True, stampa messaggi dettagliati
        save (bool): Se True, salva le immagini processate
    
    Returns:
        tuple: (spritesheet_processati, frame_totali_processati)
    """
    # Trova la directory delle immagini
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    images_dir = os.path.join(parent_dir, "dataset", "images")
    
    if not os.path.exists(images_dir):
        print(f"Errore: Directory delle immagini non trovata: {images_dir}")
        return 0, 0
    
    # Crea directory di output per le immagini con sfondo trasparente
    output_dir = os.path.join(parent_dir, "dataset", "image_transparent")
    if save:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        elif verbose:
            print(f"La directory di output esiste già: {output_dir}")
    
    # Trova tutte le cartelle spritesheet
    spritesheet_pattern = os.path.join(images_dir, "spritesheet_*")
    all_spritesheets = glob.glob(spritesheet_pattern)
    
    if verbose:
        print(f"Trovati {len(all_spritesheets)} spritesheet totali.")
    
    # Processa ogni spritesheet
    processed_spritesheets = 0
    total_frames_processed = 0
    
    for spritesheet_path in all_spritesheets:
        frames_processed = process_spritesheet_frames(
            spritesheet_path, output_dir, verbose=verbose, save=save
        )
        
        if frames_processed > 0:
            processed_spritesheets += 1
            total_frames_processed += frames_processed
    
    return processed_spritesheets, total_frames_processed

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Rimuovi lo sfondo grigio dai frame degli spritesheet.')
    parser.add_argument('--dry-run', action='store_true', help='Simula senza salvare i cambiamenti')
    parser.add_argument('-v', '--verbose', action='store_true', help='Mostra output dettagliato')
    parser.add_argument('-q', '--quiet', action='store_true', help='Esegui silenziosamente')
    parser.add_argument('--folder', help='Processa solo una specifica cartella spritesheet')
    args = parser.parse_args()
    
    verbose = args.verbose and not args.quiet
    save = not args.dry_run
    
    # Processa un singolo spritesheet se specificato
    if args.folder:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        images_dir = os.path.join(parent_dir, "dataset", "images")
        output_dir = os.path.join(parent_dir, "dataset", "image_transparent")
        
        if save and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Verifica se il path esiste, se no prova ad interpretarlo come numero
        spritesheet_path = args.folder
        if not os.path.exists(args.folder):
            folder_num = re.search(r"(\d+)", args.folder)
            if folder_num:
                spritesheet_path = os.path.join(images_dir, f"spritesheet_{folder_num.group(1)}")
        
        # Se ancora non esiste, cerca di prependervi il path corretto
        if not os.path.exists(spritesheet_path) and not spritesheet_path.startswith(os.path.join(images_dir, "spritesheet_")):
            spritesheet_path = os.path.join(images_dir, f"spritesheet_{spritesheet_path}")
        
        if os.path.exists(spritesheet_path):
            frames_processed = process_spritesheet_frames(spritesheet_path, output_dir, verbose=verbose, save=save)
            if not args.quiet:
                if args.dry_run:
                    print(f"Sarebbero stati processati {frames_processed} frame in {os.path.basename(spritesheet_path)}.")
                else:
                    print(f"Processati {frames_processed} frame in {os.path.basename(spritesheet_path)} e salvati in {output_dir}/{os.path.basename(spritesheet_path)}.")
        else:
            print(f"Errore: Spritesheet non trovato: {args.folder}")
            return
    else:
        # Processa tutti gli spritesheet
        processed_sheets, processed_frames = process_all_spritesheets(verbose=verbose, save=save)
        
        if not args.quiet:
            output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset", "image_transparent")
            if args.dry_run:
                print(f"Sarebbero stati processati {processed_frames} frame in {processed_sheets} spritesheet.")
            else:
                print(f"Processati {processed_frames} frame in {processed_sheets} spritesheet.")
                print(f"Le immagini con sfondo trasparente sono state salvate in: {output_dir}")

if __name__ == "__main__":
    main()