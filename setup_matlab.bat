@echo off
set VENV=.venv
set MATLAB_ENGINE_PATH="C:\Program Files\MATLAB\R2024b\extern\engines\python"

echo 🚀 Installation de l'API MATLAB Engine (si MATLAB est installé)...

REM Vérifie si le dossier MATLAB existe
if exist %MATLAB_ENGINE_PATH% (
    REM Active l'environnement virtuel
    call %VENV%\Scripts\activate.bat

    REM Va dans le dossier du moteur MATLAB
    cd /d %MATLAB_ENGINE_PATH%

    REM Lance l'installation
    python setup.py install
    echo ✅ MATLAB Engine installé avec succès
) else (
    echo ⚠️  Dossier MATLAB introuvable à %MATLAB_ENGINE_PATH%
    echo Veuillez vérifier que MATLAB Desktop est bien installé
)

pause
