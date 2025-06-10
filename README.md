# PDF Page Modifier

A powerful desktop application for PDF page management with dual-mode functionality. Easily remove unwanted pages or extract specific pages from PDF files through an intuitive graphical interface with visual page previews.

![image](https://github.com/user-attachments/assets/df2a1e22-1800-41bc-beee-854a930a5a50)

## Features

- **🎯 Dual-Mode Operation**: 
  - **Remove Mode**: Remove selected pages (red borders)
  - **Keep Mode**: Keep only selected pages (green borders)
- **📄 Load PDF Files**: Open any PDF file through a file browser dialog
- **🖼️ Visual Page Preview**: High-quality thumbnail previews of all PDF pages
- **🎨 Color-Coded Selection**: Visual feedback with red/green borders based on mode
- **⚡ Batch Operations**: Select all pages or clear selection with dedicated buttons
- **📊 Progress Tracking**: Real-time progress bar and status updates during operations
- **💾 Smart Export**: Save modified PDFs with intelligent operation summaries
- **🚀 Instant Access**: Open created PDFs immediately or show in file explorer
- **🔧 No External Dependencies**: Pure Python solution with no additional software required

## Requirements

- Python 3.7 or higher
- Windows 10/11 (tested on Windows 10)
- Additional dependencies will be installed via pip

## Installation

1. **Clone or download this repository** to your local machine

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   That's it! No additional system dependencies required.

## Usage

1. **Run the application**:
   ```bash
   python pdf_page_modifier.py
   ```

2. **Load a PDF file**:
   - Click the "Load PDF File" button
   - Select your PDF file from the file browser
   - Wait for page thumbnails to generate

3. **Choose your mode**:
   - **🔴 Remove Selected Pages**: Select pages to remove (red borders)
   - **🟢 Keep Selected Pages**: Select pages to keep (green borders)

4. **Select pages**:
   - Click on page thumbnails to select/deselect them
   - Selected pages show colored borders matching your chosen mode
   - Use "Select All Pages" to select all pages
   - Use "Clear Selection" to deselect all pages

5. **Save and access your PDF**:
   - Click "Save Modified PDF" button
   - Choose a location and filename for the new PDF
   - **Instantly open** the created PDF with the "📂 Open PDF" button
   - **Or show in folder** with the "📁 Show in Folder" button
  
     ![image](https://github.com/user-attachments/assets/dd8ac35a-2d1f-48b3-a29c-bb33b94cf3b9)


## Interface Overview

- **Left Panel (Controls)**:
  - **Load PDF File** button
  - **File information** display (filename and page count)
  - **Mode selection** radio buttons (Remove/Keep)
  - **Page selection information** with dynamic text based on mode
  - **Select All/Clear Selection** buttons
  - **Progress bar** and status display
  - **Save Modified PDF** button

- **Right Panel (Page Preview)**:
  - **Scrollable grid** of high-quality page thumbnails (4 columns)
  - **Click thumbnails** to toggle selection
  - **Color-coded borders**:
    - 🔴 **Red borders**: Pages to remove (Remove mode)
    - 🟢 **Green borders**: Pages to keep (Keep mode)
    - ⚪ **White background**: Unselected pages

- **Bottom**: Interactive usage instructions

- **Success Dialog** (after saving):
  - ✅ **Success confirmation** with operation summary
  - 📂 **"Open PDF"** button to instantly view the created file
  - 📁 **"Show in Folder"** button to locate the file
     - ⌨️ **Keyboard shortcuts** (Enter to open, Escape to close)

## Dual-Mode Operations

### 🔴 Remove Mode (Default)
**Use Case**: Remove unwanted pages from a PDF
- **Visual**: Selected pages have **red borders**
- **Action**: Selected pages are **removed** from the final PDF
- **Examples**:
  - Remove advertisement pages
  - Remove blank pages
  - Remove specific chapters or sections
  - Remove cover pages

### 🟢 Keep Mode
**Use Case**: Extract specific pages from a PDF
- **Visual**: Selected pages have **green borders**
- **Action**: Only selected pages are **kept** in the final PDF
- **Examples**:
  - Extract specific pages for sharing
  - Create a summary with key pages
  - Extract diagrams or charts
  - Create a custom document from multiple sources

### Mode Switching
- **Dynamic visual feedback**: Colors change automatically when switching modes
- **Preserved selection**: Your page selection remains when switching modes
- **Smart labeling**: Interface text updates to match the current mode

## Troubleshooting

### Common Issues

1. **"No module named 'fitz'" error**:
   - Make sure you've installed all requirements: `pip install -r requirements.txt`
   - PyMuPDF should install automatically with the name 'fitz'

2. **"Unable to get page count" or similar PDF errors**:
   - Ensure the PDF file is not corrupted or password-protected
   - Try with a different PDF file

3. **Application freezes when loading large PDFs**:
   - This is normal for large PDFs - the application processes pages in the background
   - Wait for the progress bar to complete

### Performance Tips

- **Large PDFs** (100+ pages) may take longer to load due to thumbnail generation
- The application uses **threading** to prevent GUI freezing during operations
- **High-quality thumbnails** are generated at 1.5x scale for better preview
- Consider closing other applications if you experience memory issues with very large PDFs
- **PyMuPDF** provides faster processing compared to traditional PDF libraries

## Technical Details

- **GUI Framework**: tkinter (Python standard library)
- **PDF Processing**: PyPDF2 for writing PDFs, PyMuPDF for reading and image generation
- **Image Processing**: Pillow (PIL) for image handling
- **PDF to Image Conversion**: PyMuPDF (fitz) - no external dependencies required
- **Threading**: Used for non-blocking PDF operations

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the application. 
