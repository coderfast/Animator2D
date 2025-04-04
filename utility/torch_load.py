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