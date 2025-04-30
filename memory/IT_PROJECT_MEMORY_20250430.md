# Memoria del Progetto: Animator2D

## Informazioni di Base
- **Nome**: Animator2D
- **Obiettivo**: Generare animazioni pixel-art sprite da descrizioni testuali per sviluppatori di giochi indie. L'utente fornisce il primo frame dell'animazione, la descrizione dell'azione, la direzione e il numero di frame, e il modello genera un'animazione (GIF, sprite sheet, video) con sprite di altezza ideale 35 pixel (range 25-50).
- **Tecnologie**: PyTorch, Transformers, Gradio, Diffusers, PIL, NumPy. GPU (CUDA) quando disponibile, CPU fallback.
- **Dataset**: 
  - Inizialmente: `pawkanarek/spraix_1024` (sprite con descrizioni, azioni, direzioni).
  - Attualmente: Dataset personalizzato `Lod34/sprite-animation` (derivato da *Gameface*), con sprite sheet tagliati in frame individuali e organizzati in cartelle. Include `sprite_metadata.json` con descrizioni e parametri, e cartella `images` con frame numerati. Sfondo rimosso dagli sprite per evitare distrazioni durante il training. Frame di background generati dal codice *Sprite Sheet Decoder* rimossi per correggere problemi di numerazione.
- **Stato attuale**: Nessuna versione completamente funzionante. *v3-alpha* su Hugging Face Spaces (`Lod34/Animator2D`) è l'ultima, con pixel simili a sprite ma animazioni incoerenti. *v1* (locale, training completato, interfaccia in sviluppo) e *v1.1* (online su Hugging Face, in sviluppo).
- **Repository GitHub**: [https://github.com/Lod34/Animator2D](https://github.com/Lod34/Animator2D)

## Versioni del Progetto
- **Animator2D-v1-alpha** (21-22 febbraio 2025): 
  - Architettura: BERT come text encoder, conv generator per sprite 64x64.
  - Interfaccia: Gradio base con output simulati (cerchi gialli su sfondo blu).
  - Output: Pixel noise incoerenti.
  - Problemi: BERT non adatto per coerenza visiva, generator troppo semplice.
- **Animator2D-mini-v1-alpha** (26 febbraio - 1 marzo 2025):
  - Architettura: CLIP come text encoder, generator leggero.
  - Varianti: 10e (10 epochs), 100e (100 epochs), 250e (250 epochs).
  - Output: Forme vaghe (10e), miglioramenti minimi (100e), parziale stabilità a 128x128 (250e), ma inutilizzabili.
  - Training: Batch size 8-16, learning rate 1e-4/2e-4.
  - Dataset: `pawkanarek/spraix_1024`.
- **Animator2D-v2-alpha** (2-3 marzo 2025):
  - Architettura: T5 come text encoder, Frame Interpolator per animazioni multi-frame, generator conv più complesso.
  - Interfaccia: Gradio avanzato.
  - Deployment: Hugging Face Spaces, problema con upload `.pth` errato ("yellow ball on blue background"), poi fixato ma animazioni incoerenti.
- **Animator2D-v3-alpha** (6 marzo 2025):
  - Architettura: T5, generator con Residual Blocks e Self-Attention per dettaglio e coerenza.
  - Training: AdamW, Cosine Annealing, 80/20 split su `pawkanarek/spraix_1024`.
  - Interfaccia: Gradio con controllo FPS e output GIF.
  - Output: Pixel sprite-like ma animazioni incoerenti.
  - Deployment: Hugging Face Spaces (`Lod34/Animator2D`).
- **Animator2D-v1** (dal 6 marzo 2025):
  - Approccio: Modulare ispirato a Da Vinci Resolve:
    1. **Creation**: Creazione o importazione di uno sprite base (es. 64x64), possibilmente con Stable Diffusion per pixel-art o decomposizione personally in parti.
    2. **Animation**: Impostazione parametri (primo frame, azione, direzione, frame) per animare lo sprite.
    3. **Generation**: Output in GIF, sprite sheet, video.
  - Training: Completato, usa il primo frame come input invece della descrizione testuale dello sprite, mantenendo azione, direzione, numero di frame.
  - Problemi:
    - Output disordinato (risolto).
    - Perdita di file salvati in Colab (necessità di salvare pesi su Google Drive ogni 5-10 epoche).
    - Canale alpha: Generazione di colori casuali nello sfondo, anche nei primi frame (che dovrebbero essere trasparenti).
    - Animazioni non generate: Frame output sono copie sfocate del primo frame, senza movimenti (es. archi, esplosioni).
  - Risultati: Training su 50 epoche con prestazioni pessime, nessuna animazione coerente.
  - Stato: Interfaccia in sviluppo, training code non funzionante.
- **Animator2D-v1.1** (dal 6 aprile 2025):
  - Versione online su Hugging Face, utilizza dataset `Lod34/sprite-animation`.
  - Obiettivo: Accessibilità online senza setup locale.
  - Stato: In sviluppo.

## Log delle Attività
- **21 febbraio 2025**: Inizio sviluppo Animator2D-v1-alpha.
- **22 febbraio 2025**: Rilascio Animator2D-v1-alpha.
- **26 febbraio 2025**: Inizio sviluppo Animator2D-mini-v1-alpha.
- **1 marzo 2025**: Rilascio Animator2D-mini-v1-alpha.
- **2 marzo 2025**: Inizio sviluppo Animator2D-v2-alpha.
- **3 marzo 2025**: Rilascio Animator2D-v2-alpha.
- **6 marzo 2025**: Rilascio Animator2D-v3-alpha e inizio sviluppo Animator2D-v1.
- **6 aprile 2025**: Inizio sviluppo Animator2D-v1.1.
- **30 aprile 2025**: Rimozione sfondo dagli sprite nel dataset e frame di background generati da *Sprite Sheet Decoder*. Completamento training code per *v1* con problemi (output disordinato risolto, perdita file in Colab, canale alpha, animazioni non generate).

## Prompt Generati
- **Prompt per continuare il lavoro**:
  ```markdown
  Sono al lavoro su un progetto per generare animazioni pixel art utilizzando un modello di deep learning chiamato Animator2D, implementato in PyTorch. Il modello prende il primo frame di una sequenza, la descrizione dell'azione, la direzione e il numero di frame, e genera i frame successivi. Il progetto è organizzato in diverse celle di codice:

  - **Cella 4**: SpriteDataset, che carica il dataset di sequenze di frame e metadati.
  - **Cella 6**: Animator2D e FrameGenerator, che definiscono il modello per la generazione dei frame.
  - **Cella 7**: PixelArtLoss, una loss personalizzata con termini per penalizzare errori nei canali RGBA, consistenza del colore, trasparenza dello sfondo e variazioni temporali (temporal_diff_weight=0.1).
  - **Cella 8**: train_model, la funzione di addestramento e valutazione.

  **Problemi Correnti**:
  - Animazioni non generate: I frame output sono copie sfocate del primo frame, senza movimenti (es. archi, esplosioni).
  - Canale alpha: Il modello genera colori casuali nello sfondo, anche nei primi frame che dovrebbero essere trasparenti.
  - Perdita di file salvati in Colab: Necessità di salvare i pesi su Google Drive ogni 5-10 epoche.
  - Output disordinato: Risolto.

  **Modifiche Implementate**:
  - Numero di Frame Visualizzati: Aggiunto un parametro `max_frames_to_visualize` per rendere configurabile il numero di frame visualizzati.
  - Trasparenza dello Sfondo: Aumentato `background_alpha_weight` a 1.0 nella PixelArtLoss e aggiunta una visualizzazione del canale alfa per debug.
  - Animazione Coerente: Aggiunto un meccanismo di attenzione nel FrameGenerator per migliorare l'uso degli embedding18 per migliorare la coerenza delle animazioni. Aggiunto anche un termine `temporal_diff_loss` nella PixelArtLoss per incoraggiare variazioni tra i frame.
  - Output Pulito: Sostituita la doppia barra di progresso con una singola barra che itera su tutti i batch, rimossi i print verbosi durante l'addestramento.
  - GIF: Aggiunta una funzione per creare GIF dai frame generati, per valutare l'animazione.
  - Dataset: Rimossi sfondi dagli sprite e frame di background generati da *Sprite Sheet Decoder* per evitare problemi di numerazione.
  - Input: Sostituita la descrizione testuale dello sprite con il primo frame dell'animazione.

  **Stato Corrente**:
  - Training su 50 epoche completato, ma risultati pessimi: nessuna animazione coerente.
  - La trasparenza dello sfondo è un problema persistente, con colori casuali anche nei primi frame.
  - L'output durante l'addestramento è ora pulito, con una singola barra di progresso che si sovrascrive correttamente.
  - Il dataset è piccolo e sbilanciato (es. per la dimensione (512, 512): 17 sequenze con 4 frame, 6 con 3 frame, 3 con 2 frame).

  **Prossimi Passaggi Suggeriti**:
  - Configurare il salvataggio dei pesi su Google Drive ogni 5-10 epoche in Colab.
  - Migliorare la gestione del canale alpha per garantire sfondi trasparenti.
  - Aumentare `temporal_diff_weight` nella PixelArtLoss a 0.3 per incoraggiare variazioni tra i frame.
  - Controllare i GIF generati (es. `epoch_X_dim_(256,256)_animation.gif`) per valutare la coerenza dell'animazione.
  - Bilanciare il dataset aggiungendo più sequenze, specialmente per lunghezze meno rappresentate (es. 2 o 3 frame).
  - Valutare un fine-tuning del TextEncoder o un'architettura più complessa (es. ConvLSTM) per migliorare la coerenza delle animazioni.

  **Richieste**:
  - Configurare il salvataggio dei pesi su Google Drive in Colab.
  - Suggerire soluzioni per il problema del canale alpha (colori casuali nello sfondo).
  - Proporre modifiche al training code per generare animazioni coerenti (es. ConvLSTM o altre architetture).
  - Ridurre la verbosità nella fase di valutazione (es. stampare i valori dei canali solo per la prima dimensione e visualizzare il canale alfa solo per il primo frame generato).
  ```

## Note Aggiuntive
- **Problemi principali**:
  - Animazioni non generate: Frame output sono copie sfocate del primo frame.
  - Canale alpha: Colori casuali nello sfondo, anche nei primi frame.
  - Perdita di file in Colab: Necessità di salvare su Google Drive.
  - Dataset limitato e sbilanciato.
- **Soluzioni implementate**:
  - Rimozione sfondi e frame di background dal dataset.
  - Output disordinato risolto.
  - Numero di frame configurabile, peso per trasparenza, meccanismo di attenzione, output pulito, creazione di GIF.
- **Prossimi passi**:
  - Configurare salvataggio su Google Drive.
  - Migliorare gestione del canale alpha.
  - Modificare il training code per generare animazioni coerenti.
  - Aumentare `temporal_diff_weight` a 0.3.
  - Analisi dei GIF generati.
  - Bilanciamento del dataset.
  - Possibile fine-tuning del TextEncoder o uso di ConvLSTM.

## Contesto Iniziale
- **Data**: 30 aprile 2025
- **Dettagli**: L'utente ha richiesto la creazione di una Mind Chat persistente per memorizzare il progetto, con una context window ampia e una struttura Markdown per organizzare la memoria. La Mind Chat deve generare prompt per Fleeting Chat temporanee dedicate a compiti specifici (es. coding, preprocessing). Discussione su problemi di lag con chat lunghe (Grok, Claude, ChatGPT) e impossibilità di usare LLaMA 4 Scout per mancanza di hardware. Raccomandazione: usare Claude 3.5 Sonnet (200k token) o Grok 3. Strategie per evitare lag: struttura modulare, capitoli per nuove conversazioni, backup su Notion/Google Docs, prompt concisi.