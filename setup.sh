#!/bin/bash

VENV=".venv"
MATLAB_ENGINE_PATH="/Applications/MATLAB_R2024b.app/extern/engines/python"
source "$VENV/bin/activate"

echo "🚀 Installation de l'API MATLAB Engine (si MATLAB est installé)..."
if [ -d "$MATLAB_ENGINE_PATH" ]; then
    cd "$MATLAB_ENGINE_PATH"
    python setup.py install
    echo "✅ MATLAB Engine installé avec succès"
else
    echo "⚠️  Dossier MATLAB introuvable à $MATLAB_ENGINE_PATH"
    echo "Veuillez vérifier que MATLAB Desktop est bien installé"
fi