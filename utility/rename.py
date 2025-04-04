import os
import math

def rename_folders_and_frames(base_dir):
    # Itera su tutte le sottocartelle nella directory base
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        
        # Controlla se è una directory
        if os.path.isdir(folder_path):
            print(f"Processing folder: {folder_name}")
            
            # Estrai il numero dalla cartella (se esiste)
            if not folder_name.startswith("spritesheet_"):
                match = folder_name.split("_")[0]
                if match.isdigit():
                    new_folder_name = f"spritesheet_{match}"
                    new_folder_path = os.path.join(base_dir, new_folder_name)
                    os.rename(folder_path, new_folder_path)
                    folder_path = new_folder_path
                    print(f"Renamed folder {folder_name} -> {new_folder_name}")
                else:
                    print(f"Skipping folder {folder_name}: no valid number found.")
                    continue
            
            # Rinominazione dei file all'interno della cartella
            frame_files = [f for f in os.listdir(folder_path) if f.endswith(".png")]
            frame_files.sort()  # Ordina i file per nome
            
            # Rinominazione dei file
            for idx, file_name in enumerate(sorted(frame_files, key=lambda x: int(x.split("_")[1].split(".")[0]))):
                new_name = f"frame_{idx}.png"  # Usa il nuovo schema
                old_path = os.path.join(folder_path, file_name)
                new_path = os.path.join(folder_path, new_name)
                os.rename(old_path, new_path)
                print(f"Renamed {file_name} -> {new_name}")

if __name__ == "__main__":
    images_dir = "/Users/lorenzo/Documents/GitHub/Animator2D/images"
    rename_folders_and_frames(images_dir)