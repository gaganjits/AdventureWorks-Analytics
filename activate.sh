#!/bin/bash
# Activation script for Mac/Linux
# Usage: source activate.sh

echo "Activating AdventureWorks virtual environment..."
source venv/bin/activate

echo ""
echo "✓ Virtual environment activated!"
echo ""
echo "Python version: $(python --version)"
echo ""
echo "Quick commands:"
echo "  - python verify_setup.py      (Verify installation)"
echo "  - jupyter notebook             (Start Jupyter)"
echo "  - pip list                     (List installed packages)"
echo "  - deactivate                   (Exit virtual environment)"
echo ""
