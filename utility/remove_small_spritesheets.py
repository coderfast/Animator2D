#!/usr/bin/env python3
"""
Remove Small Spritesheets Script
-------------------------------
Questo script rimuove dalla cartella dataset tutte le cartelle di spritesheet 
che contengono 2 o meno frame.
"""

import os
import glob
import shutil
import json
import re
from pathlib import Path

def find_and_remove_small_spritesheets(verbose=True, dry_run=False):
    """
    Trova e rimuove gli spritesheet che hanno 2 o meno frame.
    
    Args:
        verbose (bool): Se True, stampa messaggi dettagliati
        dry_run (bool): Se True, simula l'operazione senza rimuovere file
    
    Returns:
        tuple: (rimossi, totali) - numero di spritesheet rimossi e numero totale
    """
    # Trova la directory delle immagini
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    images_dir = os.path.join(parent_dir, "dataset", "images")
    
    if not os.path.exists(images_dir):
        print(f"Errore: Directory delle immagini non trovata: {images_dir}")
        return 0, 0
    
    # Directory per spritesheet rimossi (se necessaria)
    removed_dir = os.path.join(parent_dir, "dataset", "removed_spritesheets")
    if not dry_run:
        os.makedirs(removed_dir, exist_ok=True)
    
    # Trova tutte le cartelle spritesheet_*
    spritesheet_pattern = os.path.join(images_dir, "spritesheet_*")
    all_spritesheets = glob.glob(spritesheet_pattern)
    
    if verbose:
        print(f"Trovati {len(all_spritesheets)} spritesheet totali.")
    
    # Trova gli spritesheet da rimuovere
    to_remove = []
    for spritesheet_path in all_spritesheets:
        # Controlla quanti frame ci sono
        frame_pattern = os.path.join(spritesheet_path, "frame_*.png")
        frame_files = glob.glob(frame_pattern)
        frame_count = len(frame_files)
        
        folder_name = os.path.basename(spritesheet_path)
        spritesheet_num = re.search(r"spritesheet_(\d+)", folder_name)
        
        if spritesheet_num:
            spritesheet_id = spritesheet_num.group(1)
        else:
            spritesheet_id = "unknown"
        
        # Identifica spritesheet con ≤ 2 frame
        if frame_count <= 2:
            to_remove.append({
                'path': spritesheet_path,
                'id': spritesheet_id,
                'frames': frame_count,
                'folder_name': folder_name
            })
    
    # Nessuno spritesheet da rimuovere
    if not to_remove:
        if verbose:
            print("Nessuno spritesheet con 2 o meno frame trovato.")
        return 0, len(all_spritesheets)
    
    # Mostra riepilogo degli spritesheet da rimuovere
    print(f"\nSpritesheet da rimuovere ({len(to_remove)}):")
    for idx, sprite in enumerate(to_remove, 1):
        print(f"{idx}. Spritesheet {sprite['id']} - {sprite['frames']} frame")
    
    # Chiedi conferma all'utente
    if not dry_run:
        while True:
            conferma = input("\nVuoi procedere con la rimozione? (s/n): ").lower().strip()
            if conferma in ['s', 'sì', 'si', 'y', 'yes']:
                break
            elif conferma in ['n', 'no']:
                print("Operazione annullata dall'utente.")
                return 0, len(all_spritesheets)
            else:
                print("Risposta non valida. Inserisci 's' per procedere o 'n' per annullare.")
    
    # Procedi con la rimozione
    removed_count = 0
    for sprite in to_remove:
        if verbose:
            print(f"Rimozione spritesheet {sprite['id']} con {sprite['frames']} frame...")
        
        if not dry_run:
            # Sposta nella cartella dei rimossi
            try:
                dest_path = os.path.join(removed_dir, sprite['folder_name'])
                shutil.move(sprite['path'], dest_path)
                removed_count += 1
            except Exception as e:
                print(f"Errore nella rimozione di {sprite['path']}: {e}")
        else:
            removed_count += 1
    
    # Aggiorna il file sprite_registry.json se esiste
    registry_file = os.path.join(images_dir, "sprite_registry.json")
    if os.path.exists(registry_file) and not dry_run:
        try:
            with open(registry_file, 'r') as f:
                registry = json.load(f)
            
            # Trova di nuovo tutte le cartelle spritesheet rimaste
            remaining_spritesheets = glob.glob(spritesheet_pattern)
            remaining_ids = [re.search(r"spritesheet_(\d+)", os.path.basename(p)).group(1) 
                             for p in remaining_spritesheets 
                             if re.search(r"spritesheet_(\d+)", os.path.basename(p))]
            
            # Aggiorna il registro
            if 'cut' in registry:
                registry['cut'] = [id for id in registry['cut'] if id in remaining_ids]
            
            with open(registry_file, 'w') as f:
                json.dump(registry, f, indent=2)
                
            if verbose:
                print(f"File di registro aggiornato: {registry_file}")
                
        except Exception as e:
            print(f"Errore nell'aggiornamento del registro: {e}")
    
    return removed_count, len(all_spritesheets)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Rimuovi spritesheet con 2 o meno frame.')
    parser.add_argument('--dry-run', action='store_true', help='Simula senza rimuovere file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Mostra output dettagliato')
    parser.add_argument('-q', '--quiet', action='store_true', help='Esegui silenziosamente')
    args = parser.parse_args()
    
    verbose = args.verbose and not args.quiet
    
    removed, total = find_and_remove_small_spritesheets(verbose=verbose, dry_run=args.dry_run)
    
    if not args.quiet:
        if args.dry_run:
            print(f"Sarebbero stati rimossi {removed} spritesheet su {total} totali.")
        else:
            print(f"Rimossi {removed} spritesheet su {total} totali.")

if __name__ == "__main__":
    main()