import configparser
import os
import pandas as pd
from pathlib import Path

def get_config():
    # Obtenir le chemin absolu du répertoire racine du projet
    project_root = Path(__file__).parent.parent
    
    config = configparser.ConfigParser()
    config_path = os.path.join(project_root, 'config', 'config.ini')
    config.read(config_path)
    return config

def get_data(filename):
    config = get_config()
    if not filename.endswith('.csv'):
        filename += '.csv'
    file_path = os.path.join("..", config['paths']['data_folder'], filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Le fichier {filename} n'existe pas dans le dossier {config['paths']['data_folder']}.")
    return pd.read_csv(file_path, parse_dates=['Date'], index_col=0).squeeze()