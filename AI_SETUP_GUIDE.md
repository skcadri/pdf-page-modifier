# 🤖 AI-Powered Page Selection Setup Guide

## Overview

The PDF Page Modifier now includes an advanced AI-powered feature that uses computer vision to automatically select similar pages based on user-provided examples. This feature leverages a locally hosted Large Language Model (LLM) through LM Studio to analyze document pages and identify visual/structural similarities.

## Prerequisites

### 1. LM Studio Installation
- Download and install [LM Studio](https://lmstudio.ai) for your operating system
- LM Studio provides a user-friendly interface for running local LLMs

### 2. Vision-Capable Model
You need a multimodal (vision-capable) model. We recommend:
- **Gemma 3 27B** (recommended for best performance)
- **LLaVA models** (alternative option)
- **Qwen2-VL models** (alternative option)

### 3. System Requirements
- **RAM**: At least 32GB for Gemma 3 27B (16GB for smaller models)
- **GPU**: NVIDIA GPU with 12GB+ VRAM (optional but recommended)
- **CPU**: Modern multi-core processor
- **Storage**: 20GB+ free space for the model

## Setup Instructions

### Step 1: Install and Launch LM Studio
1. Download LM Studio from [https://lmstudio.ai](https://lmstudio.ai)
2. Install and launch the application
3. Allow it through your firewall if prompted

### Step 2: Download a Vision Model
1. In LM Studio, go to the **Search** tab
2. Search for `gemma-3-27b` or `llava`
3. Look for models that specifically mention "vision" or "multimodal"
4. Download a GGUF format model (recommended: Q4_K_M quantization for balance of speed/quality)

### Step 3: Load the Model
1. Go to the **Chat** tab in LM Studio
2. Select your downloaded vision model from the dropdown
3. Click **Load Model**
4. Wait for the model to load completely

### Step 4: Start the Local Server
1. Go to the **Local Server** tab in LM Studio
2. Click **Start Server**
3. Ensure the server is running on `localhost:1234`
4. The server should show "Ready" status

### Step 5: Test in PDF Page Modifier
1. Launch the PDF Page Modifier application
2. Load a PDF file
3. In the "🤖 AI-Powered Selection" section, click **Connect to LM Studio**
4. If successful, you'll see a green checkmark with the model name

## How to Use AI-Powered Selection

### 1. Select Example Pages
1. Click **📚 Select Example Pages** to enter AI mode
2. The interface will switch to blue highlighting for examples
3. Click on 1-3 pages that represent the type you want to find
4. These should be pages with similar layout, content type, or visual structure

### 2. Find Similar Pages
1. After selecting examples, click **🔍 Find Similar Pages**
2. The AI will analyze all pages in the document
3. Progress will be shown as the AI processes each page
4. Similar pages will be automatically selected

### 3. Review and Save
1. Review the AI's selection (you can manually adjust if needed)
2. Choose your mode (Remove or Keep selected pages)
3. Click **Save Modified PDF** to create your final document

## Use Cases

### Document Organization
- **Research Papers**: Select title pages, then find all other title pages
- **Reports**: Select table of contents pages, then find similar structural pages
- **Manuals**: Select instruction pages with specific layouts

### Content Filtering
- **Presentations**: Remove slide types (like agenda slides) by example
- **Forms**: Keep only certain types of form pages
- **Books**: Remove appendix pages that follow a pattern

### Quality Control
- **Scanning**: Identify and remove poorly scanned pages
- **Duplicates**: Find pages with similar content structure
- **Formatting**: Select pages with consistent formatting

## Troubleshooting

### Connection Issues
- **"Connection Failed"**: Ensure LM Studio server is running on port 1234
- **"No models found"**: Make sure a model is loaded in LM Studio's Chat tab
- **Firewall blocking**: Allow LM Studio through Windows/Mac firewall

### Performance Issues
- **Slow analysis**: Use a smaller/faster model or reduce PDF page count
- **Memory errors**: Close other applications, use lower quantization model
- **Timeout errors**: Try restarting LM Studio and reconnecting

### AI Selection Issues
- **Poor results**: Select more diverse examples or clearer visual differences
- **Too many/few matches**: Adjust your example selection strategy
- **Inconsistent results**: Use examples with very distinct visual patterns

## Model Recommendations

### Best Performance
- **Gemma 3 27B Q4_K_M**: Excellent vision capabilities, good balance
- **LLaVA 1.6 34B Q4_K_M**: Strong multimodal understanding

### Faster Options
- **LLaVA 1.5 13B Q4_K_M**: Good speed/quality balance
- **Gemma 3 9B Q4_K_M**: Faster but less accurate

### Memory-Constrained Systems
- **LLaVA 1.5 7B Q4_K_M**: Works with 16GB RAM
- **Qwen2-VL 7B Q4_K_M**: Efficient vision model

## Tips for Best Results

### Example Selection
- Choose 1-3 pages that clearly represent the pattern you want
- Select pages with distinct visual differences from what you don't want
- Use examples from different parts of the document if possible

### Document Preparation
- Ensure PDF pages are clear and readable
- Higher resolution PDFs generally give better AI analysis
- Avoid heavily compressed or low-quality scans

### Performance Optimization
- Close unnecessary applications before AI analysis
- Use smaller PDFs for testing (under 50 pages)
- Consider processing large documents in sections

## Technical Details

The AI feature works by:
1. **Page Analysis**: Each example page is analyzed by the LLM to extract visual/structural features
2. **Comparison**: Every other page is compared against the example descriptions
3. **Similarity Scoring**: The LLM determines if pages match the pattern
4. **Selection**: Pages marked as "SIMILAR" are automatically selected

The system uses OpenAI-compatible API calls to communicate with LM Studio, sending base64-encoded page images for analysis.

## Support

If you encounter issues:
1. Check the LM Studio logs for error messages
2. Try a different vision model
3. Restart both LM Studio and the PDF Page Modifier
4. Ensure your system meets the minimum requirements

For additional help, consult the LM Studio documentation at [https://lmstudio.ai/docs](https://lmstudio.ai/docs) 