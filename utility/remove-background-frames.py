#!/usr/bin/env python3
"""
Remove Background Frames Script
-------------------------------
Questo script analizza gli spritesheet e rimuove i frame che contengono solo lo sfondo.
Questi frame vengono spesso creati durante il processo di ritaglio a griglia degli spritesheet,
e generalmente sono gli ultimi frame di una sequenza.
"""

import os
import glob
import shutil
import json
import re
from pathlib import Path
from PIL import Image
import numpy as np

def is_background_frame(image_path, threshold=0.99, pixel_threshold=None):
    """
    Determina se un frame contiene solo lo sfondo basandosi sull'analisi dell'immagine.
    
    Args:
        image_path (str): Percorso all'immagine da analizzare
        threshold (float): Soglia di trasparenza/uniformità (0.0-1.0)
        pixel_threshold (int, optional): Numero minimo di pixel non-background
    
    Returns:
        bool: True se è un frame di solo sfondo, False altrimenti
    """
    try:
        with Image.open(image_path) as img:
            # Converti in RGBA se necessario
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            # Ottieni i dati dell'immagine come array
            data = np.array(img)
            
            # Controlla la trasparenza
            if img.mode == 'RGBA':
                # Conta i pixel completamente trasparenti (alpha = 0)
                transparent_pixels = np.sum(data[:, :, 3] == 0)
                total_pixels = data.shape[0] * data.shape[1]
                
                # Se la maggior parte dei pixel è trasparente
                transparent_ratio = transparent_pixels / total_pixels
                
                if transparent_ratio >= threshold:
                    return True
                
                # Se un pixel threshold è specificato, controlla anche quello
                if pixel_threshold is not None:
                    non_transparent_pixels = total_pixels - transparent_pixels
                    if non_transparent_pixels <= pixel_threshold:
                        return True
            
            # Controlla se l'immagine è COMPLETAMENTE di un solo colore
            # (per immagini senza trasparenza)
            if transparent_ratio < threshold:
                # Controlla se ci sono colori diversi dal grigio di sfondo
                # Considera il grigio di sfondo come [128, 128, 128] con una piccola tolleranza
                background_color = np.array([128, 128, 128])
                tolerance = 5  # Tolleranza più bassa per il colore di sfondo
                
                # Calcola la differenza per ogni pixel rispetto al colore di sfondo
                diff = np.abs(data[:, :, :3] - background_color).sum(axis=2)
                
                # Se esiste almeno un pixel che è significativamente diverso dal grigio di sfondo
                # allora l'immagine ha contenuto significativo
                non_grey_pixels = np.sum(diff > tolerance * 3)  # *3 perché sommiamo 3 canali
                
                if non_grey_pixels > 0:
                    return False
                
                # Solo se TUTTI i pixel sono grigi, consideriamo questo come sfondo
                return True
            
            # Se arriviamo qui, l'immagine ha contenuto significativo
            return False
    except Exception as e:
        print(f"Errore nell'analisi dell'immagine {image_path}: {e}")
        return False

def find_and_remove_background_frames(verbose=True, dry_run=False, threshold=0.99, pixel_threshold=20):
    """
    Trova e rimuove i frame che contengono solo lo sfondo.
    
    Args:
        verbose (bool): Se True, stampa messaggi dettagliati
        dry_run (bool): Se True, simula l'operazione senza rimuovere file
        threshold (float): Soglia per determinare se un frame è solo sfondo
        pixel_threshold (int): Numero minimo di pixel non-background
    
    Returns:
        tuple: (rimossi, totali) - numero di frame rimossi e numero totale
    """
    # Trova la directory delle immagini
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    images_dir = os.path.join(parent_dir, "dataset", "images")
    
    if not os.path.exists(images_dir):
        print(f"Errore: Directory delle immagini non trovata: {images_dir}")
        return 0, 0
    
    # Directory per i frame rimossi (se necessaria)
    removed_dir = os.path.join(parent_dir, "dataset", "removed_background_frames")
    if not dry_run:
        os.makedirs(removed_dir, exist_ok=True)
    
    # Trova tutti gli spritesheet
    spritesheet_pattern = os.path.join(images_dir, "spritesheet_*")
    all_spritesheets = glob.glob(spritesheet_pattern)
    
    if verbose:
        print(f"Analisi di {len(all_spritesheets)} spritesheet totali.")
    
    # Statistiche
    total_frames = 0
    to_remove = []
    frame_stats = {}
    
    # Analizza ogni spritesheet
    for spritesheet_path in all_spritesheets:
        folder_name = os.path.basename(spritesheet_path)
        spritesheet_num = re.search(r"spritesheet_(\d+)", folder_name)
        
        if spritesheet_num:
            spritesheet_id = spritesheet_num.group(1)
        else:
            spritesheet_id = "unknown"
        
        # Trova tutti i frame
        frame_pattern = os.path.join(spritesheet_path, "frame_*.png")
        frame_files = sorted(glob.glob(frame_pattern))
        
        # Conta i frame totali per questo spritesheet
        frame_count = len(frame_files)
        total_frames += frame_count
        
        if frame_count == 0:
            continue
        
        # Esamina ogni frame, in particolare gli ultimi
        background_frames = []
        
        for frame_path in frame_files:
            frame_name = os.path.basename(frame_path)
            frame_num = re.search(r"frame_(\d+)\.png", frame_name)
            
            if frame_num:
                frame_id = int(frame_num.group(1))
            else:
                continue
                
            # Analizza l'immagine per determinare se è uno sfondo
            if is_background_frame(frame_path, threshold, pixel_threshold):
                background_frames.append({
                    'path': frame_path,
                    'id': frame_id,
                    'name': frame_name,
                    'spritesheet_id': spritesheet_id
                })
        
        # Aggiorna statistiche
        if background_frames:
            frame_stats[spritesheet_id] = {
                'total': frame_count,
                'background': len(background_frames)
            }
            to_remove.extend(background_frames)
    
    # Nessun frame da rimuovere
    if not to_remove:
        if verbose:
            print("Nessun frame di solo sfondo trovato.")
        return 0, total_frames
    
    # Mostra riepilogo dei frame da rimuovere
    print(f"\nFrame di solo sfondo trovati: {len(to_remove)} su {total_frames} totali.")
    
    if verbose:
        print("\nStatistiche per spritesheet:")
        for spritesheet_id, stats in frame_stats.items():
            percentage = (stats['background'] / stats['total']) * 100
            print(f"Spritesheet {spritesheet_id}: {stats['background']} frame di sfondo su {stats['total']} ({percentage:.1f}%)")
    
    # Lista fino a 10 esempi di frame da rimuovere
    print("\nEsempi di frame da rimuovere:")
    examples = sorted(to_remove, key=lambda x: (x['spritesheet_id'], x['id']))[:10]
    for idx, frame in enumerate(examples, 1):
        print(f"{idx}. Spritesheet {frame['spritesheet_id']} - {frame['name']}")
    
    if len(to_remove) > 10:
        print(f"...e altri {len(to_remove) - 10} frame")
    
    # Chiedi conferma all'utente
    if not dry_run:
        while True:
            conferma = input("\nVuoi procedere con la rimozione? (s/n): ").lower().strip()
            if conferma in ['s', 'sì', 'si', 'y', 'yes']:
                break
            elif conferma in ['n', 'no']:
                print("Operazione annullata dall'utente.")
                return 0, total_frames
            else:
                print("Risposta non valida. Inserisci 's' per procedere o 'n' per annullare.")
    
    # Procedi con la rimozione
    removed_count = 0
    for frame in to_remove:
        if verbose:
            print(f"Rimozione frame {frame['name']} da spritesheet {frame['spritesheet_id']}...")
        
        if not dry_run:
            # Crea directory specifica dello spritesheet nella cartella dei rimossi
            spritesheet_removed_dir = os.path.join(removed_dir, f"spritesheet_{frame['spritesheet_id']}")
            os.makedirs(spritesheet_removed_dir, exist_ok=True)
            
            # Sposta il frame nella cartella dei rimossi
            try:
                dest_path = os.path.join(spritesheet_removed_dir, frame['name'])
                shutil.move(frame['path'], dest_path)
                removed_count += 1
            except Exception as e:
                print(f"Errore nella rimozione di {frame['path']}: {e}")
        else:
            removed_count += 1
    
    return removed_count, total_frames

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Rimuovi frame che contengono solo lo sfondo.')
    parser.add_argument('--dry-run', action='store_true', help='Simula senza rimuovere file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Mostra output dettagliato')
    parser.add_argument('-q', '--quiet', action='store_true', help='Esegui silenziosamente')
    parser.add_argument('-t', '--threshold', type=float, default=0.99, 
                        help='Soglia di trasparenza (0.0-1.0, default: 0.99)')
    parser.add_argument('-p', '--pixel-threshold', type=int, default=20, 
                        help='Numero minimo di pixel non-background (default: 20)')
    args = parser.parse_args()
    
    verbose = args.verbose and not args.quiet
    
    removed, total = find_and_remove_background_frames(
        verbose=verbose, 
        dry_run=args.dry_run,
        threshold=args.threshold,
        pixel_threshold=args.pixel_threshold
    )
    
    if not args.quiet:
        if args.dry_run:
            print(f"Sarebbero stati rimossi {removed} frame su {total} totali.")
        else:
            print(f"Rimossi {removed} frame su {total} totali.")

if __name__ == "__main__":
    main()