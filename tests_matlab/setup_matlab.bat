@echo off
setlocal

REM 💡 Répertoire de l’environnement virtuel
set VENV=.venv

REM 💡 Chemin du moteur MATLAB (à adapter selon ta version)
set MATLAB_ENGINE_PATH="C:\Program Files\MATLAB\R2024b\extern\engines\python"

echo 🚀 Installation de l'API MATLAB Engine (si MATLAB est installé)...

REM Vérifie si le dossier MATLAB existe
if exist %MATLAB_ENGINE_PATH% (
    REM Active l'environnement virtuel
    call %VENV%\Scripts\activate.bat

    REM Se déplace dans le répertoire de l’API MATLAB Engine
    pushd %MATLAB_ENGINE_PATH%

    REM Installe le moteur MATLAB avec le Python de l’environnement virtuel
    python setup.py install

    REM Retour au dossier initial
    popd

    echo ✅ MATLAB Engine installé dans l’environnement virtuel
) else (
    echo ❌ Dossier MATLAB introuvable à %MATLAB_ENGINE_PATH%
    echo ❗ Vérifie le chemin ou l'installation de MATLAB
)

pause
