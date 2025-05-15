function [var_results, granger_results] = estimate_VAR(data1, data2, p)
    % data1 : matrice (T x 1) de la première série temporelle
    % data2 : matrice (T x 1) de la deuxième série temporelle
    % p : ordre du modèle VAR (ici, on suppose P=1)
    
    if istable(data1)
        data1 = table2array(data1);
        disp("data1 is a table, converting to array.");
    end

    if istable(data2)
        data2 = table2array(data2);
        disp("data2 is a table, converting to array.");
    end

    % Vérifie si data1 est une ligne, et transpose si nécessaire
    if size(data1, 1) == 1
        data1 = transpose(data1);
    end

    if size(data2, 1) == 1
        data2 = transpose(data2);
    end

    % Combine les séries de données en une seule matrice
    data = [data1, data2];

    % Créer un modèle VAR(p) avec l'ordre spécifié
    model = varm(size(data, 2), p);  % ici, taille(data,2) correspond au nombre de variables
    EstModel = estimate(model, data);  % Estimation du modèle

    % Initialisation de la table de résultats
    var_results = table();

    % Matrice de covariance des résidus (pour calcul des erreurs standards)
    residuals_cov = EstModel.Covariance;

    % Parcours des équations pour chaque variable
    for i = 1:length(EstModel.AR)
        % Obtenir les coefficients estimés des retards
        coefs = EstModel.AR{i};  % Coefficients des retards

        % Calcul des erreurs standards (racine carrée des éléments diagonaux de la covariance des résidus)
        errors = sqrt(diag(residuals_cov));

        % Calcul des t-statistiques pour chaque coefficient
        t_values = coefs ./ errors;

        % Calcul des p-values à partir des t-statistiques
        p_values = 2 * (1 - tcdf(abs(t_values), size(data, 1) - p - 1));  % p-values

        % Créer un tableau pour cette variable
        var_results_i = table(coefs, errors, t_values, p_values);
        var_results_i.Properties.VariableNames = {'Estimate', 'StdError', 'tValue', 'pValue'};

        % Ajouter les résultats dans le tableau final
        var_results = [var_results; var_results_i];
    end

    % --- Test de causalité de Granger avec gctest ---
   % Test si data2 cause data1 (est-ce que data2 prédit data1 ?)
    [h12, p12, stat12] = gctest_perso(data1, data2);
    
    % Test si data1 cause data2 (est-ce que data1 prédit data2 ?)
    [h21, p21, stat21] = gctest_perso(data2, data1);
    granger_results = table( ...
        ["Y2 causes Y1"; "Y1 causes Y2"], ...
        [stat12; stat21], ...
        [p12; p21], ...
        ["No Cause"; "No Cause"], ...
        'VariableNames', {'Test', 'Statistic', 'pValue', 'Decision'} ...
    );

    alpha = 0.05;
    granger_results.Decision(granger_results.pValue < alpha) = "Cause";
end
