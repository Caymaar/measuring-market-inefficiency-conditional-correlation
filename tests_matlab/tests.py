import matlab.engine
import pandas as pd

class MatlabEngineWrapper:
    def __init__(self, matlab_script_path):
        # Démarre une instance MATLAB en arrière-plan
        self.engine = matlab.engine.start_matlab()
        
        # Ajouter le répertoire contenant les scripts MATLAB au chemin
        self.engine.addpath(matlab_script_path, nargout=0)
    
    def stop(self):
        """Arrête l'instance MATLAB."""
        self.engine.quit()
    
    def perform_adf_test(self, data, lag):
        """Effectue le test ADF en utilisant MATLAB."""
        data_matlab = self.pandas_to_matlab_table(data)
        adf_results = self.engine.performADFTest(data_matlab, float(lag))
        return adf_results
    
    def perform_arch_test(self, data, lag):
        """Effectue le test ARCH en utilisant MATLAB."""
        data_matlab = self.pandas_to_matlab_table(data)
        arch_results = self.engine.performARCHTest(data_matlab, float(lag))
        return arch_results

    def pandas_to_matlab_table(self, df):
        """Convertit une pandas DataFrame en une table MATLAB."""
        # Conversion des noms de colonnes de DataFrame en un tableau de noms
        var_names = df.columns.tolist()
        
        # Créer une liste de données MATLAB à partir de chaque colonne
        matlab_data = [matlab.double(df[col].values.tolist()) for col in var_names]
        
        # Créer une table MATLAB à partir des colonnes
        matlab_table = self.engine.table(*matlab_data)
        
        # Nous allons assigner les noms des variables directement à la table
        self.engine.eval('tableVarNames = {}', nargout=0)
        self.engine.workspace['tableVarNames'] = var_names  # Mettre les noms dans l'espace de travail MATLAB
        self.engine.eval('for i=1:numel(tableVarNames), matlab_table.Properties.VariableNames{i} = tableVarNames{i}; end', nargout=0)

        return matlab_table

def main():
    # Charger les données depuis le fichier
    inefficiency_data = pd.read_excel('inefficiency.xlsx')

    # Créer une instance de MatlabEngineWrapper en lui passant le chemin du dossier contenant les scripts MATLAB
    matlab_wrapper = MatlabEngineWrapper('c:/Users/bapdu/COMMUN/Dauphine/measuring-market-inefficiency-conditional-correlation/tests_matlab')
    
    # Effectuer le test ADF et le test ARCH
    adf_results = matlab_wrapper.perform_adf_test(inefficiency_data, 9)
    arch_results = matlab_wrapper.perform_arch_test(inefficiency_data, 12)
    
    # Afficher les résultats de ADF et ARCH
    print("Résultats ADF:", adf_results)
    print("Résultats ARCH:", arch_results)
    
    # Extraire les colonnes 's_p500' et 'ssec' de la DataFrame
    # R = inefficiency_data[['s_p500', 'ssec']]
    # print("R:", R)
    
    # Arrêter MATLAB après usage
    matlab_wrapper.stop()

if __name__ == "__main__":
    main()
