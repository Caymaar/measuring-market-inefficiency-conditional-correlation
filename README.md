## Tutoriel d'installation des dépendances

### Option 1 : Installation avec pip uniquement
1. Ouvrez votre terminal.
2. Placez-vous dans le dossier du projet :
    ```
    cd /chemin/vers/le/dossier
    ```
3. Installez le package via pip :
    ```
    pip install .
    ```

### Option 2 : Installation avec uv
1. Vérifiez si uv est installé (sinon, installez-le) :
    ```
    pip install uv
    ```
2. Placez-vous dans le dossier du projet :
    ```
    cd /chemin/vers/le/dossier
    ```
3. Générez l'environnement virtuel et installez toutes les dépendances en exécutant :
    ```
    uv sync
    ```
