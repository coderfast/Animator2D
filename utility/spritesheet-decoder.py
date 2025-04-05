# File: SpriteSheetDecoderGodotStyle.py
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import json
import time

class SpriteSheetDecoder:
    def __init__(self, root):
        self.root = root
        self.root.title("Sprite Sheet Decoder - Godot Style")
        
        # Inizializzazione delle variabili di stato
        self.image = None
        self.photo = None
        self.grid_lines = []
        self.sprite_frames = []  # Lista di frame per l'animazione (come SpriteFrames in Godot)
        self.current_frame_idx = 0
        self.is_animating = False
        self.animation_id = None
        self.zoom_factor = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 10.0
        
        # Dizionario per mappare i numeri ai percorsi dei file
        self.path_mapping = {}
        
        # Directory delle immagini
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        self.images_dir = os.path.join(parent_dir, "dataset/images")
        os.makedirs(self.images_dir, exist_ok=True)
        self.registry_file = os.path.join(self.images_dir, "sprite_registry.json")
        self.load_registry()
        
        # Pannello sinistro
        self.left_frame = tk.Frame(root)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.sheet_list = tk.Listbox(self.left_frame, width=10, selectmode=tk.SINGLE)
        self.sheet_list.pack(fill=tk.Y, expand=True)
        self.sheet_list.bind('<<ListboxSelect>>', self.load_sprite_sheet)
        
        # Pannello centrale
        self.center_frame = tk.Frame(root)
        self.center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Schede
        self.tabs_frame = tk.Frame(self.center_frame, bg='#1e1e1e')
        self.tabs_frame.pack(fill=tk.X)
        self.active_tab = tk.StringVar(value="spritesheet")
        tab_style = {'bg': '#2d2d2d', 'fg': '#cccccc', 'font': ('Arial', 10), 'padx': 15, 'pady': 5, 'relief': tk.FLAT}
        self.spritesheet_tab = tk.Label(self.tabs_frame, text="Spritesheet", **tab_style)
        self.spritesheet_tab.pack(side=tk.LEFT)
        self.spritesheet_tab.bind('<Button-1>', lambda e: self.switch_tab("spritesheet"))
        self.preview_tab = tk.Label(self.tabs_frame, text="Anteprima Animazione", **tab_style)
        self.preview_tab.pack(side=tk.LEFT)
        self.preview_tab.bind('<Button-1>', lambda e: self.switch_tab("preview"))
        tk.Frame(self.center_frame, height=1, bg='#3d3d3d').pack(fill=tk.X)
        
        # Contenuto centrale
        self.content_frame = tk.Frame(self.center_frame, bg='#1e1e1e')
        self.content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Spritesheet view
        self.spritesheet_frame = tk.Frame(self.content_frame, bg='#1e1e1e')
        
        # Controlli zoom
        self.zoom_controls = tk.Frame(self.spritesheet_frame, bg='#1e1e1e')
        self.zoom_controls.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # Stile pulsanti zoom
        zoom_btn_style = {
            'width': 3,
            'font': ('Arial', 10, 'bold'),
            'bg': '#2d2d2d',
            'fg': '#ffffff',
            'relief': tk.FLAT,
            'padx': 5,
            'pady': 2
        }
        
        # Contenitore per i pulsanti di zoom (allineato a destra)
        zoom_btns_frame = tk.Frame(self.zoom_controls, bg='#1e1e1e')
        zoom_btns_frame.pack(side=tk.RIGHT)
        
        self.zoom_out_btn = tk.Button(zoom_btns_frame, text="-", command=lambda: self.handle_zoom_btn('out'), **zoom_btn_style)
        self.zoom_out_btn.pack(side=tk.LEFT, padx=2)
        
        self.zoom_in_btn = tk.Button(zoom_btns_frame, text="+", command=lambda: self.handle_zoom_btn('in'), **zoom_btn_style)
        self.zoom_in_btn.pack(side=tk.LEFT, padx=2)
        
        self.canvas = tk.Canvas(self.spritesheet_frame, bg='#1e1e1e', highlightthickness=0)
        self.scrollbar_y = tk.Scrollbar(self.spritesheet_frame, orient="vertical", command=self.canvas.yview)
        self.scrollbar_x = tk.Scrollbar(self.spritesheet_frame, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(xscrollcommand=self.scrollbar_x.set, yscrollcommand=self.scrollbar_y.set)
        
        # Preview view (ispirato a AnimatedSprite2D)
        self.preview_frame = tk.Frame(self.content_frame, bg='#1e1e1e')
        self.preview_canvas = tk.Canvas(self.preview_frame, bg='#1e1e1e', highlightthickness=0)
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)
        self.bottom_controls = tk.Frame(self.preview_frame, bg='#1e1e1e')
        self.bottom_controls.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Miniplayer (ispirato a Godot)
        self.controls_frame = tk.Frame(self.bottom_controls, bg='#1e1e1e')
        self.controls_frame.pack(fill=tk.X, pady=10)
        control_style = {'bg': '#2d2d2d', 'fg': '#cccccc', 'activebackground': '#3d3d3d', 'activeforeground': '#ffffff'}
        self.play_button = tk.Button(self.controls_frame, text="▶", command=self.toggle_animation, width=3, font=('Arial', 10), **control_style)
        self.play_button.pack(side=tk.LEFT, padx=5)
        tk.Label(self.controls_frame, text="FPS:", font=('Arial', 10), bg='#1e1e1e', fg='#cccccc').pack(side=tk.LEFT, padx=2)
        self.fps_scale = tk.Scale(self.controls_frame, from_=1, to=60, orient=tk.HORIZONTAL, length=150, bg='#2d2d2d', fg='#cccccc', troughcolor='#3d3d3d')
        self.fps_scale.set(12)
        self.fps_scale.pack(side=tk.LEFT, padx=5)
        
        # Inizializza la vista dello spritesheet
        self.show_spritesheet_view()
        
        # Pannello destro
        self.right_frame = tk.Frame(root)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        self.params = {}
        param_font = ('Arial', 12)
        for label in ["Horizontal", "Vertical", "Separation_h", "Separation_v", "Offset_left", "Offset_right", "Offset_top", "Offset_bottom"]:
            frame = tk.Frame(self.right_frame)
            frame.pack(fill=tk.X, pady=2)
            tk.Label(frame, text=f"{label}:", font=param_font).pack(side=tk.LEFT)
            self.params[label] = tk.Entry(frame, width=5, font=param_font)
            self.params[label].pack(side=tk.LEFT, padx=5)
            self.params[label].bind('<FocusOut>', self.update_grid)
            self.params[label].bind('<KeyRelease>', self.update_grid)  # Aggiungi questo binding per l'aggiornamento in tempo reale
            
            # Aggiungi i pulsanti ▲ e ▼
            up_button = tk.Button(frame, text="▲", command=lambda l=label: self.change_param(l, 1), font=param_font, width=2)
            up_button.pack(side=tk.LEFT, padx=2)
            down_button = tk.Button(frame, text="▼", command=lambda l=label: self.change_param(l, -1), font=param_font, width=2)
            down_button.pack(side=tk.LEFT, padx=2)
        
        self.params["Horizontal"].bind('<Return>', lambda e: self.focus_next("Horizontal"))
        self.params["Vertical"].bind('<Return>', lambda e: self.focus_next("Vertical"))
        
        self.size_label = tk.Label(self.right_frame, text="Size: 0x0", font=param_font)
        self.size_label.pack(pady=5)
        
        # Bottone "Cut"
        self.cut_button = tk.Button(self.right_frame, text="Cut", command=self.cut_sprites, font=('Arial', 12, 'bold'))
        self.cut_button.pack(pady=5)
        self.cut_button.bind('<Return>', lambda e: self.cut_sprites())

        # Bottone "Come Precedente"
        self.apply_previous_button = tk.Button(self.right_frame, text="Come Precedente", command=self.apply_previous_grid, font=('Arial', 12))
        self.apply_previous_button.pack(pady=5)

        # Bottone "Delete"
        self.delete_button = tk.Button(self.right_frame, text="Delete", command=self.delete_frames, font=('Arial', 12))
        self.delete_button.pack(pady=5)
        
        self.skip_button = tk.Button(self.right_frame, text="Skip", command=self.skip_sprite, font=('Arial', 12))
        self.skip_button.pack(pady=5)
        
        # Sezione "Red Type"
        self.red_type_frame = tk.Frame(self.right_frame, bg='#2d2d2d')
        self.red_type_frame.pack(pady=10, fill=tk.X)
        tk.Label(self.red_type_frame, text="Red Type:", font=('Arial', 12), bg='#2d2d2d', fg='#ffffff').pack(anchor=tk.W)
        
        self.red_type_var = tk.StringVar(value="lowest")
        tk.Radiobutton(self.red_type_frame, text="Lowest Number", variable=self.red_type_var, value="lowest", bg='#2d2d2d', fg='#ffffff', selectcolor='#3d3d3d', font=('Arial', 10)).pack(anchor=tk.W)
        tk.Radiobutton(self.red_type_frame, text="Closest to Last Cut", variable=self.red_type_var, value="closest", bg='#2d2d2d', fg='#ffffff', selectcolor='#3d3d3d', font=('Arial', 10)).pack(anchor=tk.W)

        # Progress Bar Frame
        self.progress_frame = tk.Frame(self.right_frame, bg='#2d2d2d')
        self.progress_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        tk.Label(self.progress_frame, text="Progress:", font=('Arial', 12), bg='#2d2d2d', fg='#ffffff').pack(anchor=tk.W, pady=(0, 5))
        
        # Progress Bar Container (for visual border)
        progress_container = tk.Frame(self.progress_frame, bg='#3d3d3d', bd=1, relief=tk.SUNKEN)
        progress_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Vertical progress bar using Canvas
        self.progress_height = 200  # Fixed height for the progress bar
        self.progress_canvas = tk.Canvas(
            progress_container, 
            width=30, 
            height=self.progress_height, 
            bg='#1e1e1e', 
            highlightthickness=0
        )
        self.progress_canvas.pack(side=tk.TOP, pady=5)
        
        # Progress bar fill
        self.progress_fill = self.progress_canvas.create_rectangle(
            0, self.progress_height, 30, self.progress_height, 
            fill='#27ae60', outline=''
        )
        
        # Progress text
        self.progress_text = tk.StringVar(value="0 / 559")
        self.progress_label = tk.Label(
            self.progress_frame, 
            textvariable=self.progress_text, 
            font=('Arial', 10), 
            bg='#2d2d2d', 
            fg='#ffffff'
        )
        self.progress_label.pack(side=tk.TOP, pady=5)

        # Variabile per memorizzare i parametri della griglia precedente
        self.previous_grid_params = None
        
        # Bind degli eventi tastiera
        self.root.bind('<KeyPress>', self.on_key_press)
        
        # Binding degli eventi zoom
        self.canvas.bind('<MouseWheel>', self.handle_zoom)  # Windows e macOS
        self.canvas.bind('<Button-4>', self.handle_zoom)    # Linux scroll up
        self.canvas.bind('<Button-5>', self.handle_zoom)    # Linux scroll down

        # Carica direttamente i fogli dalle immagini all'avvio
        self.add_sheets()
        
        # Inizializza la barra di progresso dopo che tutto è stato caricato
        self.root.after(500, self.update_progress_bar)

    def load_registry(self):
        self.registry = set()
        self.skipped_registry = set()
        if os.path.exists(self.registry_file):
            with open(self.registry_file, 'r') as f:
                data = json.load(f)
                self.registry = set(data.get('cut', []))
                self.skipped_registry = set(data.get('skipped', []))
    
    def save_registry(self):
        with open(self.registry_file, 'w') as f:
            json.dump({'cut': list(self.registry), 'skipped': list(self.skipped_registry)}, f)
    
    def update_list_colors(self):
        for i in range(self.sheet_list.size()):
            number = self.sheet_list.get(i)
            self.sheet_list.itemconfig(i, {'bg': '#90EE90' if number in self.registry else '#ADD8E6' if number in self.skipped_registry else '#FFB6B6'})
    
    def add_sheets(self):
        png_files = [f for f in os.listdir(self.images_dir) if f.endswith(".png")]
        if not png_files:
            messagebox.showwarning("Attenzione", f"Nessun file PNG trovato in: {self.images_dir}")
            return
        image_files = sorted([(int(f.split(".")[0]), os.path.join(self.images_dir, f)) for f in png_files if f.split(".")[0].isdigit()])
        self.sheet_list.delete(0, tk.END)
        self.path_mapping.clear()
        for number, path in image_files:
            self.path_mapping[str(number)] = path
            self.sheet_list.insert(tk.END, str(number))
        self.update_list_colors()
        # Update progress bar after list has been populated
        self.root.after(100, self.update_progress_bar)  # Slight delay to ensure UI is ready
        self.load_next_uncut_sprite()
    
    def load_sprite_sheet(self, event):
        selection = self.sheet_list.curselection()
        if not selection:
            return
        number = self.sheet_list.get(selection[0])
        self.image = Image.open(self.path_mapping[number])
        
        # Reinizializzo gli sprite frames per il nuovo spritesheet
        self.sprite_frames = []
        self.current_frame_idx = 0
        
        # Auto-fit dell'immagine
        self.auto_fit_image()
        
        self.update_image()
        self.params["Horizontal"].delete(0, tk.END)
        self.params["Vertical"].delete(0, tk.END)
        self.params["Separation_h"].delete(0, tk.END)
        self.params["Separation_v"].delete(0, tk.END)
        self.params["Offset_left"].delete(0, tk.END)
        self.params["Offset_right"].delete(0, tk.END)
        self.params["Offset_top"].delete(0, tk.END)
        self.params["Offset_bottom"].delete(0, tk.END)
        self.params["Horizontal"].focus()
        self.is_animating = False
        self.play_button.config(text="▶")
        
        # Carica i parametri della griglia dal file di metadati
        folder = os.path.join(self.images_dir, f"spritesheet_{number}")
        metadata_file = os.path.join(folder, "metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                grid_params = metadata.get("grid", {})
                self.params["Horizontal"].insert(0, grid_params.get("columns", 0))
                self.params["Vertical"].insert(0, grid_params.get("rows", 0))
                self.params["Separation_h"].insert(0, grid_params.get("separation", {}).get("x", 0))
                self.params["Separation_v"].insert(0, grid_params.get("separation", {}).get("y", 0))
                self.params["Offset_left"].insert(0, grid_params.get("offset", {}).get("left", 0))
                self.params["Offset_right"].insert(0, grid_params.get("offset", {}).get("right", 0))
                self.params["Offset_top"].insert(0, grid_params.get("offset", {}).get("top", 0))
                self.params["Offset_bottom"].insert(0, grid_params.get("offset", {}).get("bottom", 0))
            
            # Salva i parametri della griglia corrente come "precedenti"
            self.previous_grid_params = {k: self.params[k].get() for k in self.params}
        else:
            # Se non ci sono metadati, non sovrascrivere i parametri precedenti
            if not self.previous_grid_params:
                self.previous_grid_params = {k: self.params[k].get() for k in self.params}
        
        # Aggiorna la griglia e rigenera gli sprite_frames
        self.update_grid(None)
    
    def auto_fit_image(self):
        """Calcola e imposta lo zoom ottimale per visualizzare l'intera immagine"""
        if not self.image:
            return
            
        # Ottieni le dimensioni del canvas e dell'immagine
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        image_width, image_height = self.image.size
        
        # Se il canvas non è ancora stato renderizzato, usa dimensioni di default
        if canvas_width <= 1:
            canvas_width = 800
        if canvas_height <= 1:
            canvas_height = 600
        
        # Calcola il fattore di zoom necessario per adattare l'immagine
        width_ratio = canvas_width / image_width
        height_ratio = canvas_height / image_height
        
        # Usa il rapporto più piccolo per assicurarsi che l'intera immagine sia visibile
        self.zoom_factor = min(width_ratio, height_ratio) * 0.9  # 90% per lasciare un po' di margine
        
        # Limita lo zoom tra min_zoom e max_zoom
        self.zoom_factor = min(max(self.zoom_factor, self.min_zoom), self.max_zoom)
    
    def update_image(self):
        if not self.image:
            return
        new_width = int(self.image.width * self.zoom_factor)
        new_height = int(self.image.height * self.zoom_factor)
        resized = self.image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.canvas.configure(scrollregion=(0, 0, new_width, new_height))
        self.update_grid(None)
    
    def focus_next(self, current):
        if current == "Horizontal":
            self.params["Vertical"].focus()
        elif current == "Vertical":
            self.cut_button.focus()
        self.update_grid(None)
    
    def update_grid(self, event):
        if not self.image:
            return
        try:
            params = {k: int(v.get() or 0) for k, v in self.params.items()}
        except ValueError:
            messagebox.showerror("Errore", "Inserisci solo numeri interi.")
            return
        
        w, h = self.image.size
        total_width = w - params["Offset_left"] - params["Offset_right"] - (params["Horizontal"] - 1) * params["Separation_h"]
        total_height = h - params["Offset_top"] - params["Offset_bottom"] - (params["Vertical"] - 1) * params["Separation_v"]
        sprite_w = total_width / params["Horizontal"] if params["Horizontal"] > 0 else 0
        sprite_h = total_height / params["Vertical"] if params["Vertical"] > 0 else 0
        self.size_label.config(text=f"Size: {sprite_w:.1f}x{sprite_h:.1f}")
        
        # Cancella la griglia esistente
        self.canvas.delete("grid")
        self.canvas.delete("selection")
        self.canvas.delete("cells")
        
        # Colori in stile Godot
        GRID_COLOR = "#2980b9"  # Blu chiaro per la griglia
        SELECTION_COLOR = "#27ae60"  # Verde per la selezione attiva
        CELL_COLOR = "#3498db"  # Blu più chiaro per le celle
        GRID_WIDTH = 1
        SELECTION_WIDTH = 2
        CELL_WIDTH = 1
        
        # Disegna la griglia principale
        for j in range(params["Horizontal"] + 1):
            x = params["Offset_left"] + j * (sprite_w + params["Separation_h"])
            # Limita la x alla larghezza dell'immagine
            x = min(x, w)
            x_scaled = x * self.zoom_factor
            self.canvas.create_line(
                x_scaled, 0,
                x_scaled, h * self.zoom_factor,
                fill=GRID_COLOR,
                width=GRID_WIDTH,
                tags="grid",
                dash=(4, 4)  # Linea tratteggiata
            )
        
        for i in range(params["Vertical"] + 1):
            y = params["Offset_top"] + i * (sprite_h + params["Separation_v"])
            # Limita la y all'altezza dell'immagine
            y = min(y, h)
            y_scaled = y * self.zoom_factor
            self.canvas.create_line(
                0, y_scaled,
                w * self.zoom_factor, y_scaled,
                fill=GRID_COLOR,
                width=GRID_WIDTH,
                tags="grid",
                dash=(4, 4)  # Linea tratteggiata
            )
        
        # Disegna i riquadri per tutte le celle
        for i in range(params["Vertical"]):
            for j in range(params["Horizontal"]):
                x1 = params["Offset_left"] + j * (sprite_w + params["Separation_h"])
                y1 = params["Offset_top"] + i * (sprite_h + params["Separation_v"])
                x2 = min(x1 + sprite_w, w)  # Limita x2 alla larghezza dell'immagine
                y2 = min(y1 + sprite_h, h)  # Limita y2 all'altezza dell'immagine
                
                # Rettangolo per ogni cella
                self.canvas.create_rectangle(
                    x1 * self.zoom_factor, y1 * self.zoom_factor,
                    x2 * self.zoom_factor, y2 * self.zoom_factor,
                    outline=CELL_COLOR,
                    width=CELL_WIDTH,
                    tags="cells"
                )
        
        # Evidenzia il frame corrente (se in modalità preview)
        if self.sprite_frames and hasattr(self, 'current_frame_idx'):
            current_row = self.current_frame_idx // params["Horizontal"]
            current_col = self.current_frame_idx % params["Horizontal"]
            
            x1 = params["Offset_left"] + current_col * (sprite_w + params["Separation_h"])
            y1 = params["Offset_top"] + current_row * (sprite_h + params["Separation_v"])
            x2 = min(x1 + sprite_w, w)  # Limita x2 alla larghezza dell'immagine
            y2 = min(y1 + sprite_h, h)  # Limita y2 all'altezza dell'immagine
            
            # Rettangolo di selezione
            self.canvas.create_rectangle(
                x1 * self.zoom_factor, y1 * self.zoom_factor,
                x2 * self.zoom_factor, y2 * self.zoom_factor,
                outline=SELECTION_COLOR,
                width=SELECTION_WIDTH,
                tags="selection"
            )
        
        self.update_sprite_frames(params, sprite_w, sprite_h)
    
    def update_sprite_frames(self, params, sprite_w, sprite_h):
        """Aggiorna i frame per l'animazione (simile a SpriteFrames di Godot)"""
        self.sprite_frames = []
        for i in range(params["Vertical"]):
            for j in range(params["Horizontal"]):
                x = params["Offset_left"] + j * (sprite_w + params["Separation_h"])
                y = params["Offset_top"] + i * (sprite_h + params["Separation_v"])
                frame = self.image.crop((int(x), int(y), int(x + sprite_w), int(y + sprite_h)))
                self.sprite_frames.append(ImageTk.PhotoImage(frame))
        self.current_frame_idx = 0
        if self.active_tab.get() == "preview" and self.sprite_frames:
            self.show_animation_frame()
    
    def show_animation_frame(self):
        """Mostra il frame corrente nel canvas di anteprima"""
        if not self.sprite_frames:
            return
        self.preview_canvas.delete("all")
        frame = self.sprite_frames[self.current_frame_idx]
        canvas_width, canvas_height = self.preview_canvas.winfo_width(), self.preview_canvas.winfo_height()
        if canvas_width > 1 and canvas_height > 1:
            x, y = (canvas_width - frame.width()) // 2, (canvas_height - frame.height()) // 2
            self.preview_canvas.create_image(x, y, anchor=tk.NW, image=frame)
    
    def toggle_animation(self):
        """Avvia o ferma l'animazione (ispirato a AnimatedSprite2D)"""
        if not self.sprite_frames:
            return
        self.is_animating = not self.is_animating
        self.play_button.config(text="⏸" if self.is_animating else "▶")
        if self.is_animating:
            self.animate()
    
    def animate(self):
        """Gestisce l'animazione frame per frame"""
        if not self.is_animating or not self.sprite_frames:
            return
        self.current_frame_idx = (self.current_frame_idx + 1) % len(self.sprite_frames)
        self.show_animation_frame()
        delay = int(1000 / self.fps_scale.get())
        self.root.after(delay, self.animate)
    
    def cut_sprites(self):
        if not self.image:
            return
        selection = self.sheet_list.curselection()
        if not selection:
            return
        number = self.sheet_list.get(selection[0])
        folder = os.path.join(self.images_dir, f"spritesheet_{number}")  # Usa il nuovo schema per le sottocartelle
        os.makedirs(folder, exist_ok=True)
        params = {k: int(v.get() or 0) for k, v in self.params.items()}
        w, h = self.image.size
        total_width = w - params["Offset_left"] - params["Offset_right"] - (params["Horizontal"] - 1) * params["Separation_h"]
        total_height = h - params["Offset_top"] - params["Offset_bottom"] - (params["Vertical"] - 1) * params["Separation_v"]
        sprite_w = total_width / params["Horizontal"]
        sprite_h = total_height / params["Vertical"]
        
        # Crea il file di metadati
        metadata = {
            "frames": params["Horizontal"] * params["Vertical"],
            "frame_size": {"width": int(sprite_w), "height": int(sprite_h)},
            "animation": {
                "speed": self.fps_scale.get(),
                "loop": True
            },
            "grid": {
                "columns": params["Horizontal"],
                "rows": params["Vertical"],
                "separation": {
                    "x": params["Separation_h"],
                    "y": params["Separation_v"]
                },
                "offset": {
                    "left": params["Offset_left"],
                    "right": params["Offset_right"],
                    "top": params["Offset_top"],
                    "bottom": params["Offset_bottom"]
                }
            }
        }
        
        # Salva i frame con il nuovo schema di nomi
        frame_count = 0
        for i in range(params["Vertical"]):
            for j in range(params["Horizontal"]):
                x1 = params["Offset_left"] + j * (sprite_w + params["Separation_h"])
                y1 = params["Offset_top"] + i * (sprite_h + params["Separation_v"])
                sprite = self.image.crop((int(x1), int(y1), int(x1 + sprite_w), int(y1 + sprite_h)))
                sprite.save(os.path.join(folder, f"frame_{frame_count}.png"))  # Usa il nuovo schema per i frame
                frame_count += 1
        
        # Salva i metadati
        with open(os.path.join(folder, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.registry.add(number)
        self.save_registry()
        self.update_list_colors()
        self.update_progress_bar()  # Aggiorna la barra di progresso
        self.load_next_uncut_sprite()
    
    def skip_sprite(self):
        """Salta lo sprite corrente e passa al successivo"""
        selection = self.sheet_list.curselection()
        if not selection:
            return
        number = self.sheet_list.get(selection[0])
        self.skipped_registry.add(number)
        self.save_registry()
        self.update_list_colors()
        self.load_next_uncut_sprite()
    
    def load_next_uncut_sprite(self):
        """Carica il prossimo sprite non ancora elaborato in base alla modalità selezionata"""
        red_type = self.red_type_var.get()
        current_selection = self.sheet_list.curselection()
        current_number = int(self.sheet_list.get(current_selection[0])) if current_selection else None
        
        if red_type == "lowest":
            # Seleziona lo spritesheet rosso con il numero più basso
            for i in range(self.sheet_list.size()):
                number = self.sheet_list.get(i)
                if number not in self.registry and number not in self.skipped_registry:
                    self.sheet_list.selection_clear(0, tk.END)
                    self.sheet_list.selection_set(i)
                    self.sheet_list.see(i)
                    self.load_sprite_sheet(None)
                    return
        elif red_type == "closest" and current_number is not None:
            # Seleziona lo spritesheet rosso più vicino al numero corrente
            uncut_numbers = [
                int(self.sheet_list.get(i))
                for i in range(self.sheet_list.size())
                if self.sheet_list.get(i) not in self.registry and self.sheet_list.get(i) not in self.skipped_registry
            ]
            closest_number = min(uncut_numbers, key=lambda x: abs(x - current_number), default=None)
            if closest_number is not None:
                closest_index = self.sheet_list.get(0, tk.END).index(str(closest_number))
                self.sheet_list.selection_clear(0, tk.END)
                self.sheet_list.selection_set(closest_index)
                self.sheet_list.see(closest_index)
                self.load_sprite_sheet(None)
    
    def switch_tab(self, tab_name):
        """Cambia tra la vista spritesheet e l'anteprima dell'animazione"""
        self.active_tab.set(tab_name)
        self.spritesheet_tab.configure(bg='#3d3d3d' if tab_name == "spritesheet" else '#2d2d2d')
        self.preview_tab.configure(bg='#3d3d3d' if tab_name == "preview" else '#2d2d2d')
        
        if tab_name == "spritesheet":
            self.preview_frame.pack_forget()
            self.show_spritesheet_view()
        else:
            self.spritesheet_frame.pack_forget()
            self.show_preview_view()
    
    def show_spritesheet_view(self):
        """Mostra la vista dello spritesheet con griglia e controlli"""
        self.spritesheet_frame.pack(fill=tk.BOTH, expand=True)
        self.scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.update_image()
    
    def show_preview_view(self):
        """Mostra la vista dell'anteprima dell'animazione"""
        self.preview_frame.pack(fill=tk.BOTH, expand=True)
        if self.sprite_frames:
            self.show_animation_frame()
    
    def on_key_press(self, event):
        """Gestisce gli shortcut da tastiera"""
        if event.keysym == "space":
            self.toggle_animation()
        elif event.keysym == "Left":
            self.prev_frame()
        elif event.keysym == "Right":
            self.next_frame()
        elif event.keysym == "Return" and event.state & 0x4:  # Ctrl+Enter
            self.cut_sprites()
        elif event.keysym == "Escape":
            self.skip_sprite()
        elif event.keysym == "BackSpace" and (event.state & 0x8 or event.state & 0x4):  # Command+Backspace (macOS) o Ctrl+Backspace (Windows)
            self.delete_frames()
    
    def handle_zoom_btn(self, direction):
        """Gestisce lo zoom tramite pulsanti + e -"""
        factor = 1.05 if direction == 'in' else 0.95  # Step più piccoli (5% invece di 10%)
        
        # Calcola il nuovo zoom mantenendolo entro i limiti
        new_zoom = min(max(self.zoom_factor * factor, self.min_zoom), self.max_zoom)
        
        if new_zoom != self.zoom_factor:
            # Calcola il centro del canvas visibile
            visible_center_x = self.canvas.canvasx(self.canvas.winfo_width() / 2)
            visible_center_y = self.canvas.canvasy(self.canvas.winfo_height() / 2)
            
            # Calcola il fattore di scala relativo
            scale = new_zoom / self.zoom_factor
            
            # Aggiorna il fattore di zoom
            self.zoom_factor = new_zoom
            
            # Aggiorna la vista centrando lo zoom
            self.canvas.scale('all', visible_center_x, visible_center_y, scale, scale)
            self.update_image()
            
            # Aggiorna la scrollregion
            self.canvas.configure(scrollregion=self.canvas.bbox('all'))
            
            # Mostra il livello di zoom corrente
            self.show_zoom_level()
    
    def handle_zoom(self, event):
        """Gestisce lo zoom con trackpad/mouse wheel"""
        # Determina la direzione dello zoom
        if event.num == 5 or event.delta < 0:  # Zoom out
            factor = 0.95  # Step più piccoli (5% invece di 10%)
        elif event.num == 4 or event.delta > 0:  # Zoom in
            factor = 1.05  # Step più piccoli (5% invece di 10%)
        else:
            return
            
        # Calcola il nuovo zoom mantenendolo entro i limiti
        new_zoom = min(max(self.zoom_factor * factor, self.min_zoom), self.max_zoom)
        
        # Se il nuovo zoom è diverso, applica lo zoom
        if new_zoom != self.zoom_factor:
            # Salva la posizione del mouse rispetto al canvas
            x = self.canvas.canvasx(event.x)
            y = self.canvas.canvasy(event.y)
            
            # Calcola il fattore di scala relativo
            scale = new_zoom / self.zoom_factor
            
            # Aggiorna il fattore di zoom
            self.zoom_factor = new_zoom
            
            # Aggiorna la vista
            self.canvas.scale('all', x, y, scale, scale)
            self.update_image()
            
            # Aggiorna la scrollregion
            self.canvas.configure(scrollregion=self.canvas.bbox('all'))
            
            # Mostra il livello di zoom corrente
            self.show_zoom_level()
    
    def show_zoom_level(self):
        """Mostra temporaneamente il livello di zoom"""
        # Rimuovi eventuali indicatori di zoom precedenti
        self.canvas.delete('zoom_level')
        
        # Crea un rettangolo semi-trasparente con il testo
        zoom_text = f"Zoom: {self.zoom_factor:.1f}x"
        x = 10
        y = 10
        padding = 5
        
        # Crea il testo per misurarne le dimensioni
        text_id = self.canvas.create_text(x, y, text=zoom_text, anchor='nw', fill='white', tags='zoom_level')
        bbox = self.canvas.bbox(text_id)
        
        # Crea il rettangolo di sfondo
        self.canvas.create_rectangle(
            bbox[0] - padding,
            bbox[1] - padding,
            bbox[2] + padding,
            bbox[3] + padding,
            fill='#2980b9',
            outline='',
            alpha=0.8,
            tags='zoom_level'
        )
        
        # Porta il testo in primo piano
        self.canvas.tag_raise(text_id)
        
        # Programma la rimozione dell'indicatore dopo 1.5 secondi
        self.root.after(1500, lambda: self.canvas.delete('zoom_level'))

    def delete_frames(self):
        """Elimina i frame generati per lo spritesheet selezionato"""
        selection = self.sheet_list.curselection()
        if not selection:
            return
            
        number = self.sheet_list.get(selection[0])
        folder = os.path.join(self.images_dir, f"{number}_frames")
        
        # Verifica se la cartella esiste
        if not os.path.exists(folder):
            # Se non ci sono suddivisioni e lo spritesheet è nello stato "skip",
            # rimuovi lo stato "skip" e ripristina il colore rosso
            if number in self.skipped_registry:
                self.skipped_registry.remove(number)
                self.save_registry()
                self.update_list_colors()
                messagebox.showinfo("Successo", f"Stato 'skip' rimosso dallo spritesheet {number}")
            else:
                messagebox.showwarning("Attenzione", f"Nessun frame trovato per lo spritesheet {number}")
            return
            
        # Chiedi conferma all'utente
        if not messagebox.askyesno("Conferma", f"Sei sicuro di voler eliminare tutti i frame generati per lo spritesheet {number}?"):
            return
            
        try:
            # Elimina tutti i file nella cartella
            for file in os.listdir(folder):
                file_path = os.path.join(folder, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            
            # Elimina la cartella
            os.rmdir(folder)
            
            # Rimuovi lo spritesheet dal registro
            if number in self.registry:
                self.registry.remove(number)
                self.save_registry()
                self.update_list_colors()
                
            messagebox.showinfo("Successo", f"Frame eliminati con successo per lo spritesheet {number}")
            
        except Exception as e:
            messagebox.showerror("Errore", f"Errore durante l'eliminazione dei frame: {str(e)}")

    def change_param(self, label, delta):
        """Incrementa o decrementa il valore del parametro specificato"""
        try:
            current_value = int(self.params[label].get() or 0)
        except ValueError:
            current_value = 0
        new_value = current_value + delta
        self.params[label].delete(0, tk.END)
        self.params[label].insert(0, str(new_value))
        self.update_grid(None)

    def apply_previous_grid(self):
        """Applica i parametri della griglia usati nello spritesheet precedente"""
        if not self.previous_grid_params:
            messagebox.showwarning("Attenzione", "Non ci sono parametri precedenti da applicare.")
            return
        
        for key, value in self.previous_grid_params.items():
            self.params[key].delete(0, tk.END)
            self.params[key].insert(0, value)
        
        self.update_grid(None)

    def update_progress_bar(self):
        """Aggiorna la barra di progresso verticale."""
        try:
            # Conta le sottocartelle "spritesheet_*"
            spritesheet_folders = [d for d in os.listdir(self.images_dir) 
                                  if os.path.isdir(os.path.join(self.images_dir, d)) 
                                  and d.startswith("spritesheet_")]
            
            cut_count = len(spritesheet_folders)
            total_count = 559  # Valore fisso come richiesto
            
            # Aggiorna il testo
            self.progress_text.set(f"{cut_count} / {total_count}")
            
            # Calcola la percentuale di completamento
            if total_count > 0:
                percentage = min(cut_count / total_count, 1.0)
            else:
                percentage = 0
                
            # Aggiorna la barra di progresso (cresce dal basso verso l'alto)
            fill_height = int(self.progress_height * percentage)
            self.progress_canvas.coords(
                self.progress_fill, 
                0, self.progress_height - fill_height, 
                30, self.progress_height
            )
            
            # Colora la barra in base alla percentuale
            if percentage < 0.33:
                color = '#e74c3c'  # Rosso per progresso basso
            elif percentage < 0.66:
                color = '#f39c12'  # Arancione per progresso medio
            else:
                color = '#27ae60'  # Verde per progresso alto
                
            self.progress_canvas.itemconfig(self.progress_fill, fill=color)
        except Exception as e:
            print(f"Error updating progress bar: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    root.configure(bg='#2d2d2d')
    app = SpriteSheetDecoder(root)
    root.mainloop()