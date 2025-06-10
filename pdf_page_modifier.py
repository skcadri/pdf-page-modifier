import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import threading
import io
import subprocess
import platform
import base64
import json
import numpy as np
from PIL import Image, ImageTk
import PyPDF2
import fitz  # PyMuPDF
import tempfile
import requests
from openai import OpenAI


class PDFPageModifier:
    def __init__(self, root):
        self.root = root
        self.root.title("PDF Page Modifier")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize variables
        self.pdf_path = None
        self.pdf_reader = None
        self.page_images = []
        self.page_thumbnails = []
        self.selected_pages = set()
        self.total_pages = 0
        self.mode = "remove"  # "remove" or "keep"
        
        # LLM Integration variables
        self.lm_studio_url = "http://127.0.0.1:1234"
        self.openai_client = None
        self.model_name = None  # Store the actual model name from LM Studio
        self.example_pages = set()  # Pages selected as examples for AI similarity
        self.ai_mode = False  # When True, show example selection UI
        
        # Create the GUI
        self.create_widgets()
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="PDF Page Modifier", 
                               font=("Arial", 18, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Control panel (left side)
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), 
                          padx=(0, 10))
        
        # Load PDF button
        self.load_btn = ttk.Button(control_frame, text="Load PDF File", 
                                  command=self.load_pdf)
        self.load_btn.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), 
                          pady=(0, 10))
        
        # File info
        self.file_info_label = ttk.Label(control_frame, text="No file loaded", 
                                        wraplength=200)
        self.file_info_label.grid(row=1, column=0, columnspan=2, 
                                 sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Mode selection
        mode_frame = ttk.LabelFrame(control_frame, text="Mode", padding="5")
        mode_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), 
                       pady=(0, 10))
        
        self.mode_var = tk.StringVar(value="remove")
        
        self.remove_mode_radio = ttk.Radiobutton(mode_frame, 
                                               text="Remove Selected Pages", 
                                               variable=self.mode_var, 
                                               value="remove",
                                               command=self.on_mode_change)
        self.remove_mode_radio.grid(row=0, column=0, sticky=tk.W, pady=2)
        
        self.keep_mode_radio = ttk.Radiobutton(mode_frame, 
                                             text="Keep Selected Pages", 
                                             variable=self.mode_var, 
                                             value="keep",
                                             command=self.on_mode_change)
        self.keep_mode_radio.grid(row=1, column=0, sticky=tk.W, pady=2)
        
        # Page selection info
        self.selection_info = ttk.Label(control_frame, 
                                       text="Pages to remove: None selected")
        self.selection_info.grid(row=3, column=0, columnspan=2, 
                                sticky=(tk.W, tk.E), pady=(0, 10))
        
        # AI-Powered Page Selection Section
        ai_frame = ttk.LabelFrame(control_frame, text="🤖 AI-Powered Selection", padding="5")
        ai_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), 
                     pady=(0, 10))
        
        # LM Studio connection status
        self.lm_status_label = ttk.Label(ai_frame, text="LM Studio: Not connected", 
                                        foreground="red")
        self.lm_status_label.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))
        
        # Connect to LM Studio button
        self.connect_lm_btn = ttk.Button(ai_frame, text="Connect to LM Studio", 
                                        command=self.connect_to_lm_studio)
        self.connect_lm_btn.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))
        
        # AI Mode toggle
        self.ai_mode_btn = ttk.Button(ai_frame, text="📚 Select Example Pages", 
                                     command=self.toggle_ai_mode, 
                                     state="disabled")
        self.ai_mode_btn.grid(row=2, column=0, sticky=(tk.W, tk.E), padx=(0, 2), pady=(0, 5))
        
        # Find similar pages button
        self.find_similar_btn = ttk.Button(ai_frame, text="🔍 Find Similar Pages", 
                                          command=self.find_similar_pages, 
                                          state="disabled")
        self.find_similar_btn.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(2, 0), pady=(0, 5))
        
        # Example pages info
        self.example_info_label = ttk.Label(ai_frame, text="Select 1-3 example pages first", 
                                           font=("Arial", 8))
        self.example_info_label.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))
        
        # Selection buttons
        self.select_all_btn = ttk.Button(control_frame, text="Select All Pages", 
                                        command=self.select_all_pages, 
                                        state="disabled")
        self.select_all_btn.grid(row=5, column=0, sticky=(tk.W, tk.E), 
                                padx=(0, 5), pady=(0, 5))
        
        self.clear_selection_btn = ttk.Button(control_frame, text="Clear Selection", 
                                            command=self.clear_selection, 
                                            state="disabled")
        self.clear_selection_btn.grid(row=5, column=1, sticky=(tk.W, tk.E), 
                                     pady=(0, 5))
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var, 
                                           maximum=100)
        self.progress_bar.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), 
                              pady=(10, 5))
        
        # Status label
        self.status_label = ttk.Label(control_frame, text="Ready", 
                                     foreground="green")
        self.status_label.grid(row=7, column=0, columnspan=2, 
                              sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Save button
        self.save_btn = ttk.Button(control_frame, text="Save Modified PDF", 
                                  command=self.save_pdf, state="disabled")
        self.save_btn.grid(row=8, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        # Page preview area (right side)
        preview_frame = ttk.LabelFrame(main_frame, text="Page Preview", padding="10")
        preview_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        
        # Create canvas with scrollbars for page thumbnails
        canvas_frame = ttk.Frame(preview_frame)
        canvas_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        
        self.canvas = tk.Canvas(canvas_frame, bg="white")
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", 
                                   command=self.canvas.yview)
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient="horizontal", 
                                   command=self.canvas.xview)
        
        self.canvas.configure(yscrollcommand=v_scrollbar.set, 
                             xscrollcommand=h_scrollbar.set)
        
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Scrollable frame inside canvas
        self.scrollable_frame = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window((0, 0), 
                                                      window=self.scrollable_frame, 
                                                      anchor="nw")
        
        # Bind canvas resize
        self.canvas.bind('<Configure>', self.on_canvas_configure)
        self.scrollable_frame.bind('<Configure>', self.on_frame_configure)
        
        # Instructions
        instructions_text = """Instructions:
1. Click 'Load PDF File' to select a PDF
2. Choose mode: 'Remove Selected Pages' (red border) or 'Keep Selected Pages' (green border)
3. Manual Selection: Click on page thumbnails to select pages
4. AI-Powered Selection: 
   • Connect to LM Studio (requires local Gemma 3 27B model)
   • Select 1-3 example pages, then click 'Find Similar Pages'
   • AI will automatically select pages similar to your examples
5. Use 'Select All' or 'Clear Selection' buttons as needed
6. Click 'Save Modified PDF' to create a new PDF based on your selection and mode"""
        
        instructions_label = ttk.Label(main_frame, text=instructions_text, 
                                      justify=tk.LEFT, font=("Arial", 9))
        instructions_label.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), 
                               pady=(10, 0))
        
    def on_canvas_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
    def on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
    def on_mode_change(self):
        """Handle mode change between remove and keep"""
        self.mode = self.mode_var.get()
        self.update_visual_feedback()
        self.update_selection_info()
        
    def load_pdf(self):
        file_path = filedialog.askopenfilename(
            title="Select PDF File",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        
        if file_path:
            self.pdf_path = file_path
            # Load PDF in a separate thread to prevent GUI freezing
            threading.Thread(target=self.process_pdf, daemon=True).start()
            
    def process_pdf(self):
        try:
            self.update_status("Loading PDF...", "orange")
            self.progress_var.set(10)
            
            # Open PDF with PyMuPDF
            self.pdf_document = fitz.open(self.pdf_path)
            self.total_pages = len(self.pdf_document)
            
            # Also open with PyPDF2 for writing (we'll keep this for the save functionality)
            with open(self.pdf_path, 'rb') as file:
                self.pdf_reader = PyPDF2.PdfReader(file)
            
            self.progress_var.set(30)
            
            # Update file info
            file_name = os.path.basename(self.pdf_path)
            self.file_info_label.config(text=f"File: {file_name}\nPages: {self.total_pages}")
            
            # Convert PDF pages to images for preview using PyMuPDF
            self.update_status("Generating page previews...", "orange")
            self.page_images = []
            
            for page_num in range(self.total_pages):
                # Get page
                page = self.pdf_document[page_num]
                
                # Convert page to image (pixmap)
                mat = fitz.Matrix(1.5, 1.5)  # Scale factor for better quality
                pixmap = page.get_pixmap(matrix=mat)
                
                # Convert pixmap to PIL Image
                img_data = pixmap.tobytes("ppm")
                img = Image.open(io.BytesIO(img_data))
                self.page_images.append(img)
                
                # Update progress
                progress = 30 + (page_num + 1) / self.total_pages * 40
                self.progress_var.set(progress)
            
            self.progress_var.set(70)
            
            # Create thumbnails
            self.create_page_thumbnails()
            
            self.progress_var.set(100)
            self.update_status("PDF loaded successfully!", "green")
            
            # Enable buttons
            self.select_all_btn.config(state="normal")
            self.clear_selection_btn.config(state="normal")
            self.save_btn.config(state="normal")
            
        except Exception as e:
            self.update_status(f"Error loading PDF: {str(e)}", "red")
            messagebox.showerror("Error", f"Failed to load PDF: {str(e)}")
            

            
    def create_page_thumbnails(self):
        # Clear existing thumbnails
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        self.page_thumbnails = []
        
        # Create thumbnail for each page
        cols = 4  # Number of columns
        for i, image in enumerate(self.page_images):
            # Resize image for thumbnail
            image.thumbnail((200, 280), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            
            row = i // cols
            col = i % cols
            
            # Create frame for each page
            page_frame = ttk.Frame(self.scrollable_frame, relief="solid", borderwidth=2)
            page_frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
            
            # Page number label
            page_label = ttk.Label(page_frame, text=f"Page {i+1}", 
                                  font=("Arial", 10, "bold"))
            page_label.grid(row=0, column=0, pady=(5, 0))
            
            # Thumbnail button
            thumb_btn = tk.Button(page_frame, image=photo, 
                                 command=lambda idx=i: self.toggle_page_selection(idx),
                                 relief="flat", borderwidth=2)
            thumb_btn.grid(row=1, column=0, padx=5, pady=5)
            thumb_btn.image = photo  # Keep a reference
            
            self.page_thumbnails.append((page_frame, thumb_btn))
            
        # Update scroll region
        self.scrollable_frame.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
        # Apply initial visual feedback
        self.update_visual_feedback()
        
    def toggle_page_selection(self, page_idx):
        if self.ai_mode:
            # In AI mode, toggle example pages
            if page_idx in self.example_pages:
                self.example_pages.remove(page_idx)
            else:
                # Limit to 3 example pages for better performance
                if len(self.example_pages) < 3:
                    self.example_pages.add(page_idx)
                else:
                    messagebox.showwarning("Limit Reached", 
                                         "Maximum 3 example pages allowed for optimal AI performance.")
                    return
            self.update_example_info()
        else:
            # Normal mode, toggle selected pages
            if page_idx in self.selected_pages:
                self.selected_pages.remove(page_idx)
            else:
                self.selected_pages.add(page_idx)
            self.update_selection_info()
            
        self.update_visual_feedback()
        
    def update_visual_feedback(self):
        """Update visual feedback based on current mode and selection"""
        if not self.page_thumbnails:
            return
            
        for i, (frame, btn) in enumerate(self.page_thumbnails):
            if self.ai_mode:
                # In AI mode, show example pages with blue border
                if i in self.example_pages:
                    btn.config(bg="lightblue", relief="solid", borderwidth=3)
                else:
                    btn.config(bg="white", relief="flat", borderwidth=2)
            else:
                # Normal mode, show selected pages with mode color
                color = "green" if self.mode == "keep" else "red"
                if i in self.selected_pages:
                    btn.config(bg=color, relief="solid", borderwidth=3)
                else:
                    btn.config(bg="white", relief="flat", borderwidth=2)
        
    def select_all_pages(self):
        self.selected_pages = set(range(self.total_pages))
        self.update_visual_feedback()
        self.update_selection_info()
        
    def clear_selection(self):
        self.selected_pages.clear()
        self.update_visual_feedback()
        self.update_selection_info()
        
    def update_selection_info(self):
        if self.selected_pages:
            pages_list = sorted(list(self.selected_pages))
            pages_str = ", ".join([str(p+1) for p in pages_list])
            
            if self.mode == "keep":
                self.selection_info.config(text=f"Pages to keep: {pages_str}")
            else:
                self.selection_info.config(text=f"Pages to remove: {pages_str}")
        else:
            if self.mode == "keep":
                self.selection_info.config(text="Pages to keep: None selected")
            else:
                self.selection_info.config(text="Pages to remove: None selected")
            
    def save_pdf(self):
        if not self.selected_pages:
            action = "keep" if self.mode == "keep" else "remove"
            messagebox.showwarning("Warning", f"No pages selected to {action}!")
            return
            
        # Ask for save location
        save_path = filedialog.asksaveasfilename(
            title="Save Modified PDF",
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")]
        )
        
        if save_path:
            threading.Thread(target=self.create_modified_pdf, 
                           args=(save_path,), daemon=True).start()
            
    def create_modified_pdf(self, save_path):
        try:
            self.update_status("Creating modified PDF...", "orange")
            self.progress_var.set(0)
            
            # Create new PDF writer
            pdf_writer = PyPDF2.PdfWriter()
            
            # Determine which pages to keep based on mode
            if self.mode == "keep":
                # Keep selected pages, remove others
                pages_to_keep = sorted(list(self.selected_pages))
                operation_text = f"Kept {len(pages_to_keep)} selected pages.\n" \
                               f"Removed {self.total_pages - len(pages_to_keep)} pages."
            else:
                # Remove selected pages, keep others
                pages_to_keep = [i for i in range(self.total_pages) 
                               if i not in self.selected_pages]
                operation_text = f"Removed {len(self.selected_pages)} selected pages.\n" \
                               f"Kept {len(pages_to_keep)} pages."
            
            with open(self.pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for i, page_idx in enumerate(pages_to_keep):
                    pdf_writer.add_page(pdf_reader.pages[page_idx])
                    progress = (i + 1) / len(pages_to_keep) * 100
                    self.progress_var.set(progress)
                    
            # Write the modified PDF
            with open(save_path, 'wb') as output_file:
                pdf_writer.write(output_file)
                
            self.progress_var.set(100)
            self.update_status("PDF saved successfully!", "green")
            
            # Show success dialog with option to open PDF
            self.show_success_dialog(save_path, operation_text, len(pages_to_keep))
            
        except Exception as e:
            self.update_status(f"Error saving PDF: {str(e)}", "red")
            messagebox.showerror("Error", f"Failed to save PDF: {str(e)}")
            
    def show_success_dialog(self, pdf_path, operation_text, page_count):
        """Show success dialog with option to open the created PDF"""
        # Create custom dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("PDF Created Successfully")
        dialog.geometry("400x200")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        
        # Success icon and message
        main_frame = ttk.Frame(dialog, padding="20")
        main_frame.pack(fill="both", expand=True)
        
        # Success message
        success_label = ttk.Label(main_frame, 
            text="✅ PDF Created Successfully!", 
            font=("Arial", 12, "bold"),
            foreground="green")
        success_label.pack(pady=(0, 10))
        
        # Details
        details_label = ttk.Label(main_frame, 
            text=f"{operation_text}\nNew PDF has {page_count} pages.",
            justify="center")
        details_label.pack(pady=(0, 20))
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=(0, 10))
        
        # Open PDF button
        open_btn = ttk.Button(button_frame, 
            text="📂 Open PDF", 
            command=lambda: self.open_pdf_file(pdf_path, dialog))
        open_btn.pack(side="left", padx=(0, 10))
        
        # Show in folder button
        folder_btn = ttk.Button(button_frame, 
            text="📁 Show in Folder", 
            command=lambda: self.show_in_folder(pdf_path, dialog))
        folder_btn.pack(side="left", padx=(0, 10))
        
        # Close button
        close_btn = ttk.Button(button_frame, 
            text="✅ Close", 
            command=dialog.destroy)
        close_btn.pack(side="left")
        
        # Focus on Open PDF button
        open_btn.focus_set()
        
        # Handle Enter key
        dialog.bind('<Return>', lambda e: self.open_pdf_file(pdf_path, dialog))
        dialog.bind('<Escape>', lambda e: dialog.destroy())
        
    def open_pdf_file(self, pdf_path, dialog=None):
        """Open the PDF file with the default application"""
        try:
            system = platform.system()
            
            if system == "Windows":
                os.startfile(pdf_path)
            elif system == "Darwin":  # macOS
                subprocess.run(["open", pdf_path])
            else:  # Linux and other Unix-like systems
                subprocess.run(["xdg-open", pdf_path])
                
            if dialog:
                dialog.destroy()
                
        except Exception as e:
            messagebox.showerror("Error", f"Could not open PDF file:\n{str(e)}")
            
    def show_in_folder(self, pdf_path, dialog=None):
        """Show the PDF file in the file explorer"""
        try:
            system = platform.system()
            
            if system == "Windows":
                subprocess.run(["explorer", "/select,", pdf_path])
            elif system == "Darwin":  # macOS
                subprocess.run(["open", "-R", pdf_path])
            else:  # Linux
                # Open the containing folder
                folder_path = os.path.dirname(pdf_path)
                subprocess.run(["xdg-open", folder_path])
                
            if dialog:
                dialog.destroy()
                
        except Exception as e:
            messagebox.showerror("Error", f"Could not show file in folder:\n{str(e)}")
            
    def update_status(self, message, color="black"):
        self.status_label.config(text=message, foreground=color)
        self.root.update_idletasks()
    
    # LLM Integration Methods
    def connect_to_lm_studio(self):
        """Connect to LM Studio and test the connection"""
        try:
            self.update_status("Connecting to LM Studio...", "orange")
            
            # Initialize OpenAI client with LM Studio URL
            self.openai_client = OpenAI(
                base_url=f"{self.lm_studio_url}/v1",
                api_key="lm-studio"  # LM Studio doesn't require a real API key
            )
            
            # Test connection by listing models
            models = self.openai_client.models.list()
            if models.data:
                # Store the actual model name for use in API calls
                self.model_name = models.data[0].id
                self.lm_status_label.config(text=f"✅ Connected to: {self.model_name}", 
                                          foreground="green")
                self.update_status("Connected to LM Studio successfully!", "green")
                
                # Enable AI features
                self.ai_mode_btn.config(state="normal")
                self.connect_lm_btn.config(text="Reconnect to LM Studio")
                
                messagebox.showinfo("Success", 
                    f"Successfully connected to LM Studio!\n"
                    f"Model: {self.model_name}\n"
                    f"You can now use AI-powered page selection.")
            else:
                raise Exception("No models found in LM Studio")
                
        except Exception as e:
            self.lm_status_label.config(text="❌ Connection failed", foreground="red")
            self.update_status("Failed to connect to LM Studio", "red")
            messagebox.showerror("Connection Failed", 
                f"Could not connect to LM Studio at {self.lm_studio_url}\n\n"
                f"Error: {str(e)}\n\n"
                f"Please ensure:\n"
                f"• LM Studio is running\n"
                f"• A vision-capable model (like Gemma 3 27B) is loaded\n"
                f"• The local server is started on port 1234")
    
    def toggle_ai_mode(self):
        """Toggle between normal selection and AI example selection mode"""
        if not self.openai_client:
            messagebox.showwarning("Not Connected", 
                "Please connect to LM Studio first before using AI features.")
            return
            
        self.ai_mode = not self.ai_mode
        
        if self.ai_mode:
            # Entering AI mode
            self.ai_mode_btn.config(text="📝 Normal Selection")
            self.example_pages.clear()
            self.update_example_info()
            messagebox.showinfo("AI Mode", 
                "AI Example Mode activated!\n\n"
                "Click on 1-3 pages that represent the type of pages you want to find.\n"
                "Then click 'Find Similar Pages' to let AI select similar pages automatically.")
        else:
            # Exiting AI mode
            self.ai_mode_btn.config(text="📚 Select Example Pages")
            self.example_pages.clear()
            self.update_example_info()
        
        self.update_visual_feedback()
        self.update_find_similar_button()
    
    def update_example_info(self):
        """Update the example pages information display"""
        if self.example_pages:
            pages_list = sorted(list(self.example_pages))
            pages_str = ", ".join([str(p+1) for p in pages_list])
            self.example_info_label.config(text=f"Example pages: {pages_str}")
        else:
            self.example_info_label.config(text="Select 1-3 example pages first")
        
        self.update_find_similar_button()
    
    def update_find_similar_button(self):
        """Enable/disable find similar button based on example selection"""
        if self.ai_mode and self.example_pages and self.openai_client:
            self.find_similar_btn.config(state="normal")
        else:
            self.find_similar_btn.config(state="disabled")
    
    def image_to_base64(self, image):
        """Convert PIL Image to base64 string for API"""
        buffer = io.BytesIO()
        # Resize image to reduce API payload size
        image_copy = image.copy()
        image_copy.thumbnail((800, 1200), Image.Resampling.LANCZOS)
        image_copy.save(buffer, format='JPEG', quality=85)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str
    
    def analyze_page_with_llm(self, page_image):
        """Analyze a single page using the LLM to extract features"""
        try:
            # Convert image to base64
            image_base64 = self.image_to_base64(page_image)
            
            # Prepare the prompt for page analysis
            prompt = """Analyze this document page image and describe its key visual characteristics in 2-3 sentences. Focus on:
- Layout: single/multi-column, headers, spacing
- Content: text density, images, lists, tables
- Style: font sizes, formatting, visual elements

Be concise and focus on patterns that help identify similar pages."""

            # Use LM Studio compatible format
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                            }
                        ]
                    }
                ],
                max_tokens=200,
                temperature=0.1
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            # Show error in GUI status
            self.update_status(f"AI Error: {str(e)}", "red")
            return None
    
    def compare_pages_with_llm(self, example_descriptions, candidate_image):
        """Compare a candidate page with example pages using LLM"""
        try:
            # Convert candidate image to base64
            candidate_base64 = self.image_to_base64(candidate_image)
            
            # Create comparison prompt
            example_text = "\n".join([f"Example {i+1}: {desc}" for i, desc in enumerate(example_descriptions)])
            
            prompt = f"""Compare this page to the examples:

{example_text}

Does this page have similar layout, content type, and visual style? Answer only "YES" or "NO"."""

            # Use LM Studio compatible format
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{candidate_base64}"}
                            }
                        ]
                    }
                ],
                max_tokens=10,
                temperature=0.0
            )
            
            result = response.choices[0].message.content.strip().upper()
            return "YES" in result
            
        except Exception as e:
            self.update_status(f"AI Compare Error: {str(e)}", "red")
            return False
    
    def find_similar_pages_worker(self, progress_callback, result_callback):
        """Worker thread for AI analysis to prevent GUI freezing"""
        try:
            # Analyze example pages first
            example_descriptions = []
            total_operations = len(self.example_pages) + (self.total_pages - len(self.example_pages))
            current_operation = 0
            
            progress_callback("Analyzing example pages...", 0)
            
            for page_idx in sorted(self.example_pages):
                description = self.analyze_page_with_llm(self.page_images[page_idx])
                if description:
                    example_descriptions.append(description)
                else:
                    result_callback(False, "Failed to analyze example pages. Check LM Studio connection.")
                    return
                    
                current_operation += 1
                progress = (current_operation / total_operations) * 100
                progress_callback("Analyzing example pages...", progress)
            
            if not example_descriptions:
                result_callback(False, "No example descriptions generated.")
                return
            
            # Compare all other pages with examples
            similar_pages = set(self.example_pages)  # Include the examples themselves
            progress_callback("Finding similar pages...", (current_operation / total_operations) * 100)
            
            for page_idx in range(self.total_pages):
                if page_idx not in self.example_pages:
                    is_similar = self.compare_pages_with_llm(
                        example_descriptions, 
                        self.page_images[page_idx]
                    )
                    
                    if is_similar:
                        similar_pages.add(page_idx)
                    
                    current_operation += 1
                    progress = (current_operation / total_operations) * 100
                    progress_callback("Finding similar pages...", progress)
            
            result_callback(True, similar_pages)
            
        except Exception as e:
            result_callback(False, f"AI analysis error: {str(e)}")
    
    def find_similar_pages(self):
        """Use LLM to find pages similar to the selected examples"""
        if not self.example_pages:
            messagebox.showwarning("No Examples", "Please select 1-3 example pages first.")
            return
            
        if not self.openai_client:
            messagebox.showwarning("Not Connected", "Please connect to LM Studio first.")
            return
        
        if not hasattr(self, 'model_name'):
            messagebox.showerror("Model Error", "No model name detected. Please reconnect to LM Studio.")
            return
        
        # Disable the button during processing
        self.find_similar_btn.config(state="disabled")
        
        def progress_callback(status, progress):
            """Update GUI progress from worker thread"""
            self.root.after(0, lambda: self.update_status(status, "orange"))
            self.root.after(0, lambda: self.progress_var.set(progress))
        
        def result_callback(success, result):
            """Handle results from worker thread"""
            def update_gui():
                if success:
                    # Exit AI mode and apply the results
                    self.ai_mode = False
                    self.ai_mode_btn.config(text="📚 Select Example Pages")
                    self.example_pages.clear()
                    self.update_example_info()
                    
                    # Apply the similar pages as selection
                    self.selected_pages = result
                    self.update_visual_feedback()
                    self.update_selection_info()
                    
                    self.progress_var.set(100)
                    self.update_status("AI analysis completed!", "green")
                    
                    # Show results
                    messagebox.showinfo("AI Analysis Complete", 
                        f"Found {len(result)} similar pages out of {self.total_pages} total pages.\n\n"
                        f"The AI has automatically selected pages with similar visual structure "
                        f"and content patterns to your examples.\n\n"
                        f"You can now review the selection and save your modified PDF.")
                else:
                    self.update_status("AI analysis failed", "red")
                    messagebox.showerror("Analysis Error", result)
                
                # Re-enable the button
                self.find_similar_btn.config(state="normal")
            
            self.root.after(0, update_gui)
        
        # Start worker thread
        worker_thread = threading.Thread(
            target=self.find_similar_pages_worker,
            args=(progress_callback, result_callback),
            daemon=True
        )
        worker_thread.start()


def main():
    root = tk.Tk()
    app = PDFPageModifier(root)
    root.mainloop()


if __name__ == "__main__":
    main() 