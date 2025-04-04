import torch
from transformers import BertModel
import torch.nn as nn
import os

# Definizione del modello (estratta da training-code.py)
class Animator2DModel(nn.Module):
    def __init__(self):
        super(Animator2DModel, self).__init__()
        print("Inizializzazione del modello...")

        # Text Encoder (BERT)
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')

        # Image Encoder (CNN)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(start_dim=1),
            nn.Linear(128 * 16 * 16, 128)
        )

        # Decoder (LSTM)
        self.decoder = nn.LSTM(input_size=768 + 128, hidden_size=256, num_layers=2, batch_first=True)

        # Frame Generator (Linear Layer)
        self.frame_generator = nn.Linear(256, 64 * 64 * 3)

        print("Modello inizializzato.")

# Percorso del file .pth
pth_path = "/Users/lorenzo/Documents/GitHub/Animator2D/Animator2D-v1.0.0/Animator2D-v1.0.0.pth"  # Cambia con il percorso del tuo file
bin_path = "/Users/lorenzo/Documents/GitHub/Animator2D/Animator2D-v1.0.0/modello_huggingface"  # Dove verrà salvato il modello

# Controlla se il file esiste
if not os.path.exists(pth_path):
    raise FileNotFoundError(f"File {pth_path} non trovato!")

# Caricamento dello stato dei pesi
state_dict = torch.load(pth_path, map_location=torch.device("cpu"))

# Inizializzazione del modello
model = Animator2DModel()

# Caricamento dei pesi nel modello
model.load_state_dict(state_dict, strict=False)  # strict=False per evitare errori di chiavi mancanti

# Salvataggio nel formato Hugging Face
os.makedirs(bin_path, exist_ok=True)
torch.save(model.state_dict(), os.path.join(bin_path, "pytorch_model.bin"))

print(f"✅ Conversione completata! Il modello è stato salvato in: {bin_path}")
