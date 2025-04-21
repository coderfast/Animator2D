#!/usr/bin/env python3
"""
Torch Model Loader Script
-------------------------
Questo script è un utility per scaricare e verificare il modello Animator2D da Hugging Face Hub.
Scarica il file del modello PyTorch (.pth) dal repository ufficiale di Animator2D
e stampa la struttura delle chiavi per verificare l'integrità del modello.
Utile sia per il debug che per controllare che il modello sia stato scaricato correttamente.
"""

import torch
from huggingface_hub import hf_hub_download

# Scarica il file localmente
model_path = hf_hub_download(
    repo_id="Lod34/Animator2D-v1.0.0",
    filename="Animator2D-v1.0.0.pth"
)

# Carica il file
state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
print(state_dict.keys())