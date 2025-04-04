import os

def remove_empty_folders(base_dir):
    # Itera su tutte le sottocartelle nella directory base
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        
        # Controlla se è una directory e se è vuota
        if os.path.isdir(folder_path) and not os.listdir(folder_path):
            os.rmdir(folder_path)
            print(f"Removed empty folder: {folder_name}")

if __name__ == "__main__":
    images_dir = "/Users/lorenzo/Documents/GitHub/Animator2D/dataset/images"
    remove_empty_folders(images_dir)