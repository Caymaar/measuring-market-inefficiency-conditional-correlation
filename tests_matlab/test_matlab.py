import pandas as pd
import matlab.engine

# 1. Charger les données depuis un fichier CSV ou Excel
df = pd.read_excel('inefficiency.xlsx', index_col=0)  # index_col=0 pour ignorer les dates

# 2. Convertir en format MATLAB
def pandas_to_matlab_table(df):
    import matlab
    matlab_data = {}
    for col in df.columns:
        matlab_data[col] = matlab.double(df[col].fillna(0).tolist())
    return matlab.table(**matlab_data)

# 3. Démarrer MATLAB
eng = matlab.engine.start_matlab()

# 4. Convertir DataFrame en table MATLAB
ml_table = pandas_to_matlab_table(df)

# 5. Ajouter le dossier MATLAB contenant les scripts/fonctions
eng.addpath(r'C:\chemin\vers\tes_scripts_matlab', nargout=0)

# 6. Appeler les fonctions MATLAB
adf_results = eng.performADFTest(ml_table, 9)
arch_results = eng.performARCHTest(ml_table, 12)

# 7. Sauvegarder les résultats dans des fichiers depuis MATLAB
eng.writetable(adf_results, 'ADFresults.xlsx', nargout=0)
eng.writetable(arch_results, 'ARCHresults.xlsx', nargout=0)

# 8. Fermer MATLAB
eng.quit()
