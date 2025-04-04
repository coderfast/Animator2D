import json
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import torchvision.transforms as transforms

# Imposta il dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilizzo dispositivo: {device}")

# Percorsi locali
metadata_path = '/Users/lorenzo/Documents/GitHub/Animator2D/sprite_metadata.json'
images_dir = '/Users/lorenzo/Documents/GitHub/Animator2D/images'

# Funzione per caricare i metadati
def load_metadata(metadata_path):
    print(f"Caricamento metadati da: {metadata_path}")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    print(f"Metadati caricati, numero di animazioni: {len(metadata)}")
    return metadata

# Funzione per verificare se un frame è uno sfondo
def is_background(frame_path):
    try:
        img = Image.open(frame_path).convert('RGB')
        img_array = np.array(img)
        return np.all(img_array == img_array[0, 0], axis=(0, 1)).all()
    except Exception as e:
        print(f"Errore nel caricamento di {frame_path}: {e}")
        return True  # Considera l'errore come sfondo per saltare il file

# Funzione per caricare i frame validi di un’animazione
def load_frames(folder_path, expected_frames):
    print(f"Caricamento frame da: {folder_path}, frame attesi: {expected_frames}")
    frames = []
    for frame_file in sorted(os.listdir(folder_path)):
        frame_path = os.path.join(folder_path, frame_file)
        if (frame_file.lower().endswith(('.png', '.jpg', '.jpeg')) and 
            frame_file != 'metadata.json'):
            if not is_background(frame_path) and len(frames) < expected_frames:
                frames.append(frame_path)
    print(f"Frame caricati: {len(frames)}")
    while len(frames) < expected_frames and len(frames) > 0:
        frames.append(frames[-1])
    return frames

# Trasformazioni per le immagini
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Funzione di collazione personalizzata
def custom_collate(batch):
    if not batch:
        print("Batch vuoto!")
        return None
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    base_frame = torch.stack([item['base_frame'] for item in batch])
    target_frames = [item['target_frames'] for item in batch]
    expected_frames = torch.tensor([item['expected_frames'] for item in batch], dtype=torch.long)
    print(f"Batch processato, dimensione: {len(batch)}")
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'base_frame': base_frame,
        'target_frames': target_frames,
        'expected_frames': expected_frames
    }

# Dataset personalizzato
class SpriteAnimationDataset(Dataset):
    def __init__(self, metadata, images_dir, transform=None):
        self.metadata = metadata
        self.images_dir = images_dir
        self.transform = transform
        self.animations = []
        
        print("Inizializzazione del dataset...")
        for anim_id, anim in metadata.items():
            folder_path = os.path.join(images_dir, anim['folder_name'])
            if os.path.exists(folder_path):
                expected_frames = int(anim['frames'])
                frames = load_frames(folder_path, expected_frames)
                if len(frames) > 0:
                    self.animations.append({
                        'description': anim['full_description'],
                        'base_frame': frames[0],
                        'target_frames': frames[1:] if len(frames) > 1 else [],
                        'expected_frames': expected_frames
                    })
        print(f"Dataset inizializzato, numero di animazioni: {len(self.animations)}")
    
    def __len__(self):
        return len(self.animations)
    
    def __getitem__(self, idx):
        anim = self.animations[idx]
        base_frame = Image.open(anim['base_frame']).convert('RGB')
        target_frames = [Image.open(frame).convert('RGB') for frame in anim['target_frames']]
        
        if self.transform:
            base_frame = self.transform(base_frame)
            target_frames = [self.transform(frame) for frame in target_frames]
        
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        inputs = tokenizer(anim['description'], return_tensors='pt', padding='max_length', 
                           truncation=True, max_length=512)
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'base_frame': base_frame,
            'target_frames': target_frames,
            'expected_frames': anim['expected_frames']
        }

# Modello Animator2D (semplificato)
class Animator2DModel(torch.nn.Module):
    def __init__(self):
        super(Animator2DModel, self).__init__()
        print("Inizializzazione del modello...")
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.image_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(128 * 16 * 16, 128)
        )
        self.decoder = torch.nn.LSTM(input_size=768 + 128, hidden_size=256, num_layers=2, batch_first=True)
        self.frame_generator = torch.nn.Linear(256, 64 * 64 * 3)
        print("Modello inizializzato")

    def forward(self, input_ids, attention_mask, base_frame, expected_frames):
        batch_size = input_ids.size(0)
        text_features = self.text_encoder(input_ids, attention_mask=attention_mask).pooler_output
        image_features = self.image_encoder(base_frame)
        combined_features = torch.cat((text_features, image_features), dim=1)
        
        generated_frames = []
        for i in range(batch_size):
            num_frames = expected_frames[i].item() - 1
            if num_frames > 0:
                combined_features_i = combined_features[i].unsqueeze(0).repeat(num_frames, 1)
                output, _ = self.decoder(combined_features_i)
                frames_i = self.frame_generator(output)
                generated_frames.append(frames_i.view(num_frames, 3, 64, 64))
            else:
                generated_frames.append(torch.empty(0, 3, 64, 64).to(input_ids.device))
        
        return generated_frames

# Funzione di training
def train_model(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        print(f"Elaborazione batch {batch_idx + 1}/{len(dataloader)}")
        if batch is None:
            print("Batch nullo, salto...")
            continue
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        base_frame = batch['base_frame'].to(device)
        target_frames = [[tf.to(device) for tf in tf_list] for tf_list in batch['target_frames']]
        expected_frames = batch['expected_frames'].to(device)
        
        optimizer.zero_grad()
        generated_frames = model(input_ids, attention_mask, base_frame, expected_frames)
        
        loss = 0
        for i in range(len(target_frames)):
            num_frames = expected_frames[i].item() - 1
            if num_frames > 0 and len(target_frames[i]) >= num_frames:
                target = torch.stack(target_frames[i][:num_frames])
                generated = generated_frames[i][:num_frames]
                loss += torch.nn.functional.mse_loss(generated, target)
        
        if len(target_frames) > 0 and loss > 0:
            loss = loss / len(target_frames)
        else:
            loss = torch.tensor(0.0, device=device)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(dataloader)

# Main
if __name__ == '__main__':
    print("Avvio del programma...")
    # Carica i metadati locali
    metadata = load_metadata(metadata_path)
    
    # Crea dataset e dataloader
    dataset = SpriteAnimationDataset(metadata, images_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=custom_collate)
    
    # Inizializza modello e optimizer
    model = Animator2DModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Ciclo di training
    for epoch in range(10):
        print(f"Inizio epoca {epoch + 1}/10")
        avg_loss = train_model(model, dataloader, optimizer, device)
        print(f'Epoca {epoch + 1}/10, Loss: {avg_loss:.4f}')
    
    # Salva il modello
    torch.save(model.state_dict(), 'animator2d_v1_0_0.pth')
    print("Modello salvato come 'animator2d_v1_0_0.pth'")