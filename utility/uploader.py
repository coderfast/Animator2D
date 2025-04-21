#!/usr/bin/env python3
"""
Dataset Uploader Script
----------------------
Questo script prepara il dataset di Animator2D per l'addestramento, copiando
le immagini di sprite dalla cartella 'images' alla cartella 'train',
mantenendo la struttura delle sottocartelle. È utile per preparare il dataset
per l'addestramento in un formato standardizzato, garantendo che i file siano
organizzati correttamente.
"""

import os
import shutil

# Percorso della cartella sorgente
images_dir = "/Users/lorenzo/Documents/GitHub/Animator2D/dataset/image_transparent"

# Percorso della cartella destinazione
destination_dir = "/Users/lorenzo/Documents/GitHub/Animator2D/dataset"

# Controlla se la cartella sorgente esiste
if not os.path.exists(images_dir):
    raise FileNotFoundError(f"La cartella {images_dir} non esiste. Verifica il percorso.")

# Controlla se la cartella destinazione esiste, altrimenti creala
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# Funzione per raccogliere i percorsi dei file PNG dalle sottocartelle
def collect_images(images_dir):
    images = []
    for subdir in os.listdir(images_dir):
        subdir_path = os.path.join(images_dir, subdir)
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                if file.endswith(".png"):
                    images.append(os.path.join(subdir_path, file))
    return images

# Raccogli i percorsi delle immagini
images = collect_images(images_dir)

# Controlla se ci sono immagini da copiare
if not images:
    raise ValueError("Nessuna immagine PNG trovata nelle sottocartelle di 'images'. Verifica la struttura della cartella.")

# Copia le immagini nella nuova destinazione
for image_path in images:
    # Crea il percorso relativo mantenendo la struttura delle sottocartelle
    relative_path = os.path.relpath(image_path, images_dir)
    destination_path = os.path.join(destination_dir, "train", relative_path)
    
    # Crea le sottocartelle necessarie
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    
    # Copia il file
    shutil.copy2(image_path, destination_path)

print(f"Immagini copiate con successo in {destination_dir}")