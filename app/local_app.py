import gradio as gr
import torch
from transformers import BertTokenizer, BertModel
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import logging

# Configura il logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Inizio inizializzazione dell'app")

# Imposta il dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Dispositivo selezionato: {device}")

# Trasformazioni per le immagini
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Definizione del modello Animator2D
class Animator2DModel(torch.nn.Module):
    def __init__(self):
        super(Animator2DModel, self).__init__()
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.image_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # Layer aggiunto
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # Layer aggiunto per corrispondere a image_encoder.6
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),  # Layer aggiunto per corrispondere a image_encoder.7
            torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # Layer aggiunto per corrispondere a image_encoder.8
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # Layer aggiunto per corrispondere a image_encoder.10
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)  # Layer finale
        )
        self.decoder = torch.nn.LSTM(input_size=768 + 128, hidden_size=256, num_layers=2, batch_first=True)
        self.frame_generator = torch.nn.Linear(256, 64 * 64 * 3)
    
    def forward(self, input_ids, attention_mask, base_frame, num_frames):
        text_features = self.text_encoder(input_ids, attention_mask=attention_mask).pooler_output
        image_features = self.image_encoder(base_frame).flatten(start_dim=1)
        combined_features = torch.cat((text_features, image_features), dim=1)
        combined_features = combined_features.unsqueeze(1).repeat(1, num_frames, 1)
        output, _ = self.decoder(combined_features)
        generated_frames = self.frame_generator(output)
        return generated_frames.view(-1, num_frames, 3, 64, 64)

# Funzione per generare i frame
def generate_animation(description, base_frame_image, num_frames):
    logger.info("Inizio generazione animazione")
    try:
        # Carica il modello da file locale
        model = Animator2DModel().to(device)
        logger.info("Modello inizializzato, caricamento pesi da file locale...")
        local_weights_path = "/Users/lorenzo/Documents/GitHub/Animator2D/Animator2D-v1.0.0/Animator2D-v1.0.0.pth"
        try:
            model.load_state_dict(torch.load(local_weights_path, map_location=device))
            logger.info("Pesi caricati correttamente.")
        except Exception as e:
            logger.error(f"Errore durante il caricamento dei pesi: {e}")
            raise
        model.eval()
        logger.info("Modello caricato con successo")

        # Prepara il testo
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        inputs = tokenizer(description, return_tensors='pt', padding='max_length', 
                          truncation=True, max_length=512)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        # Prepara l'immagine di base
        base_frame = transform(base_frame_image).unsqueeze(0).to(device)

        # Genera i frame
        with torch.no_grad():
            generated_frames = model(input_ids, attention_mask, base_frame, num_frames)
        
        # Converte i frame generati in immagini PIL
        generated_frames = generated_frames.squeeze(0).cpu().numpy()
        output_frames = []
        for i in range(num_frames):
            frame = generated_frames[i].transpose(1, 2, 0)  # Da (C, H, W) a (H, W, C)
            frame = np.clip(frame, 0, 1)  # Normalizza tra 0 e 1
            frame = (frame * 255).astype(np.uint8)  # Converte in formato immagine
            output_frames.append(Image.fromarray(frame))

        logger.info("Animazione generata con successo")
        return output_frames
    except Exception as e:
        logger.error(f"Errore durante la generazione: {str(e)}")
        raise

# Interfaccia Gradio
logger.info("Inizio configurazione interfaccia Gradio")
with gr.Blocks(title="Animator2D-v1.0.0") as demo:
    gr.Markdown("# Animator2D-v1.0.0\nInserisci una descrizione e un'immagine di base per generare un'animazione!")
    
    with gr.Row():
        with gr.Column():
            description_input = gr.Textbox(label="Descrizione dell'animazione", placeholder="Es: 'A character jumping'")
            base_frame_input = gr.Image(label="Immagine di base", type="pil")
            num_frames_input = gr.Slider(1, 5, value=3, step=1, label="Numero di frame")
            submit_button = gr.Button("Genera Animazione")
        
        with gr.Column():
            output_gallery = gr.Gallery(label="Frame generati", show_label=True)

    submit_button.click(
        fn=generate_animation,
        inputs=[description_input, base_frame_input, num_frames_input],
        outputs=output_gallery
    )

logger.info("Interfaccia Gradio configurata, avvio...")
demo.launch(server_name="0.0.0.0", server_port=7860)