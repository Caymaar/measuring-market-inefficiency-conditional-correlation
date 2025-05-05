import matlab.engine

# Démarrer une session MATLAB
eng = matlab.engine.start_matlab()

# Afficher la version de MATLAB
print(f"Version MATLAB : {eng.version()}")

# Effectuer une opération simple
result = eng.eval('2 + 3')
print(f"Le résultat de 2 + 3 est : {result}")

# Quitter MATLAB
eng.quit()
