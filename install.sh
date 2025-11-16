#!/usr/bin/env bash
set -e

echo "🔍 Checking available Python/pip commands..."

# Detect Python
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ No Python found. Please install Python 3.8+ and try again."
    exit 1
fi

# Detect pip
if command -v pip3 &>/dev/null; then
    PIP_CMD="pip3"
elif command -v pip &>/dev/null; then
    PIP_CMD="pip"
else
    echo "❌ No pip found. Please install pip and try again."
    exit 1
fi

echo "✔ Using Python: $PYTHON_CMD"
echo "✔ Using Pip: $PIP_CMD"

echo "📦 Installing requirements..."
$PIP_CMD install --upgrade pip setuptools wheel
$PIP_CMD install -r requirements.txt

echo "🎉 Installation complete!"
