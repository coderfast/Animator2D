#!/usr/bin/env python3
import os
import json
import sys
import argparse

def update_registry(force=False, verbose=False):
    """
    Updates the sprite_registry.json file by checking for spritesheet_* directories
    to determine which spritesheets have been cut.
    
    Args:
        force (bool): If True, automatically remove missing directories from registry
        verbose (bool): If True, print more detailed information
    """
    # Find the images directory (same logic as in the spritesheet-decoder.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    images_dir = os.path.join(parent_dir, "dataset/images")
    
    if not os.path.exists(images_dir):
        print(f"Error: Images directory not found at {images_dir}")
        return False
    
    registry_file = os.path.join(images_dir, "sprite_registry.json")
    
    # Load existing registry if it exists
    existing_registry = {'cut': [], 'skipped': []}
    if os.path.exists(registry_file):
        try:
            with open(registry_file, 'r') as f:
                existing_registry = json.load(f)
        except Exception as e:
            print(f"Warning: Error loading existing registry: {e}")
            print("Creating new registry...")
    
    # Make sure all required keys exist
    if 'cut' not in existing_registry:
        existing_registry['cut'] = []
    if 'skipped' not in existing_registry:
        existing_registry['skipped'] = []
    
    # Find all spritesheet_* directories - simplified to just check directory existence
    cut_spritesheets = set()
    for item in os.listdir(images_dir):
        item_path = os.path.join(images_dir, item)
        if os.path.isdir(item_path) and item.startswith("spritesheet_"):
            try:
                # Extract the number from spritesheet_123 format
                number = item.split("_")[1]
                cut_spritesheets.add(number)
                if verbose:
                    print(f"Found spritesheet directory: {item}")
            except Exception as e:
                print(f"Warning: Failed to process directory {item}: {e}")
    
    # Convert lists to sets for easier comparison
    existing_cut = set(existing_registry['cut'])
    existing_skipped = set(existing_registry['skipped'])
    
    # Check for differences
    newly_cut = cut_spritesheets - existing_cut
    missing_cut = existing_cut - cut_spritesheets
    
    if newly_cut:
        print(f"Found {len(newly_cut)} newly cut spritesheets: {sorted(newly_cut)}")
    
    if missing_cut:
        print(f"Warning: {len(missing_cut)} spritesheets are in the registry but missing directories: {sorted(missing_cut)}")
        if force:
            print("Force mode enabled, automatically removing from registry")
            choice = 'y'
        else:
            choice = input("Remove these from the registry? (y/n): ").lower()
        
        if choice == 'y':
            for number in missing_cut:
                if number in existing_cut:
                    existing_cut.remove(number)
    
    # Get all PNG files to check for unprocessed files
    all_png_files = set()
    for item in os.listdir(images_dir):
        if item.endswith(".png") and os.path.isfile(os.path.join(images_dir, item)):
            try:
                number = item.split(".")[0]
                if number.isdigit():
                    all_png_files.add(number)
            except Exception:
                pass
    
    # Find unprocessed PNGs (not in cut or skipped)
    unprocessed = all_png_files - (cut_spritesheets | existing_skipped)
    if unprocessed:
        if verbose:
            print(f"Found {len(unprocessed)} unprocessed PNG files: {sorted(unprocessed)}")
        else:
            print(f"Found {len(unprocessed)} unprocessed PNG files")
    
    # Update the registry
    existing_registry['cut'] = sorted(list(cut_spritesheets), key=lambda x: int(x) if x.isdigit() else x)
    
    # Save the updated registry
    try:
        with open(registry_file, 'w') as f:
            json.dump(existing_registry, f, indent=2)
        print(f"Registry updated successfully at {registry_file}")
        print(f"Cut spritesheets: {len(existing_registry['cut'])}")
        print(f"Skipped spritesheets: {len(existing_registry['skipped'])}")
        print(f"Unprocessed spritesheets: {len(unprocessed)}")
        
        # Calculate percentage complete
        total_sheets = len(all_png_files)
        if total_sheets > 0:
            processed = len(cut_spritesheets) + len(existing_skipped)
            percentage = (processed / total_sheets) * 100
            print(f"Progress: {percentage:.1f}% complete ({processed}/{total_sheets})")
            
        return True
    except Exception as e:
        print(f"Error: Failed to save registry: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Update the sprite registry by scanning directories')
    parser.add_argument('--force', '-f', action='store_true', help='Automatically remove missing directories from registry')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print more detailed information')
    
    args = parser.parse_args()
    success = update_registry(force=args.force, verbose=args.verbose)
    sys.exit(0 if success else 1)
