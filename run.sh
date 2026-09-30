#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "🚀 Initializing Iris-Image Pipeline..."

# 1. Locate and activate virtual environment
# Prefer shared QT venv (has all deps pre-installed), fallback to local .venv
SHARED_VENV="/mnt/sda5/Projects/QT/.venv"
LOCAL_VENV=".venv"

if [ -d "$SHARED_VENV" ]; then
    echo "🔄 Using shared Q-Ternary virtual environment..."
    source "$SHARED_VENV/bin/activate"
elif [ -d "$LOCAL_VENV" ]; then
    echo "🔄 Using local virtual environment..."
    source "$LOCAL_VENV/bin/activate"
else
    echo "📦 Creating Python virtual environment..."
    python3 -m venv "$LOCAL_VENV"
    source "$LOCAL_VENV/bin/activate"
    echo "📥 Installing dependencies (this may take a moment)..."
    pip install --upgrade pip
    pip install -r requirements.txt
fi

# 2. Run the hybrid quantum machine learning engine
echo "⚡ Running Hybrid QML Engine..."
echo "==============================================="
echo "⏳ Loading Heavy Scientific Libraries (PyTorch, PennyLane, XGBoost)..."
mkdir -p outputs models
export PYTHONUNBUFFERED=1
python3 src/main.py --dataset-path mnist --n-features 12 --n-layers 3 --epochs 15 --max-ram 6.0 --run-name "MNIST Direct Encoding" 2>&1 | tee outputs/training_log.txt

echo -e "\n📊 Generating Performance Graphs automatically..."
export PYTHONPATH="$(pwd)/src"
python3 test/generate_graphs.py --run-dir models/latest
