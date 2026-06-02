#!/usr/bin/env bash
set -e

echo "Creating Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

if [ ! -f .env ]; then
  cp .env.example .env
fi

echo "Setup complete."
echo "Next steps:"
echo "1) Install Ollama from https://ollama.com"
echo "2) Run: ollama pull llama3.2:3b"
echo "3) Run: streamlit run streamlit_app.py"
