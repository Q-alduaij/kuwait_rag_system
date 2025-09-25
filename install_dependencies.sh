#!/bin/bash

echo "🚀 Installing Kuwait RAG System Dependencies..."
echo "=================================================="

# Update pip first
pip install --upgrade pip

# Install base requirements
echo "📦 Installing core dependencies..."
pip install -r requirements.txt

# Install system-specific dependencies for Arabic NLP
echo "🔤 Installing Arabic NLP dependencies..."

# For Ubuntu/Debian systems
if command -v apt-get &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y python3-dev libxml2-dev libxslt1-dev zlib1g-dev
    sudo apt-get install -y tesseract-ocr tesseract-ocr-ara
    sudo apt-get install -y libsm6 libxext6 libxrender-dev
fi

# For CentOS/RHEL systems
if command -v yum &> /dev/null; then
    sudo yum install -y python3-devel libxml2-devel libxslt-devel zlib-devel
    sudo yum install -y tesseract tesseract-langpack-ara
fi

# Install CAMeL Tools for advanced Arabic NLP (optional)
echo "🐫 Installing CAMeL Tools for advanced Arabic NLP..."
pip install camel-tools

# Download CAMeL Tools models
echo "📥 Downloading Arabic language models..."
python -c "
try:
    from camel_tools.utils.download import Downloader
    d = Downloader()
    d.download('all')
    print('✅ Arabic models downloaded successfully')
except Exception as e:
    print('⚠️  Could not download Arabic models:', e)
"

# Test critical installations
echo "🧪 Testing critical installations..."
python -c "
import sys
try:
    import langchain, chromadb, transformers, torch
    import arabic_reshaper, pyarabic
    print('✅ Core dependencies installed successfully')
except ImportError as e:
    print('❌ Missing dependency:', e)
    sys.exit(1)
"

echo "🎉 Installation completed successfully!"
echo "=================================================="
echo "Next steps:"
echo "1. Organize your data in data/raw/"
echo "2. Run: python run_processing.py"
echo "3. Start API: uvicorn main:app --reload"
