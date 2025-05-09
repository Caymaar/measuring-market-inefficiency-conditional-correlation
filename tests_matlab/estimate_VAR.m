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

    % --- Test de Granger avec test de Wald ---
    % X : matrice des variables indépendantes
    X = data(1:end-1, :); % On utilise les données de t-1 comme exogènes
    Y = data(2:end, :);   % On utilise les données de t comme dépendantes

    % Estimation des coefficients VAR
    B = EstModel.AR{1};  % Coefficients du modèle VAR
    Sigma = EstModel.Covariance;  % Matrice de covariance des erreurs
    numCoeff = size(B, 2);  % Nombre de coefficients par équation
    k = size(B, 1);  % Nombre de variables

    % --- Test de Granger Y1 ⇒ Y2 ---
    % Matrice de contrainte R1 pour le test Y2 ⇒ Y1 (test si X2 cause Y1)
    R1 = zeros(p, k * numCoeff); 
    for i = 1:p
        col = 1 + (i - 1) * k + 2;  % colonne de X2 (2e variable)
        R1(i, col) = 1;
    end
    q1 = zeros(p, 1);  % Le vecteur des contraintes

    % Calcul du test de Wald Y2 ⇒ Y1
    B_vec = B(:);  % Vectorisation des coefficients estimés
    wald_stat1 = (R1 * B_vec - q1)' / (R1 * (kron(inv(X' * X), Sigma)) * R1') * (R1 * B_vec - q1);
    pval1 = 1 - chi2cdf(wald_stat1, p);  % Calcul de la p-value

    % --- Test de Granger Y2 ⇒ Y1 ---
    % Matrice de contrainte R2 pour le test Y1 ⇒ Y2 (test si X1 cause Y2)
    R2 = zeros(p, k * numCoeff);
    for i = 1:p
        col = numCoeff + 1 + (i - 1) * k + 1;  % colonne de X1 pour eq. Y2
        R2(i, col) = 1;
    end
    q2 = zeros(p, 1);  % Le vecteur des contraintes

    % Calcul du test de Wald Y1 ⇒ Y2
    wald_stat2 = (R2 * B_vec - q2)' / (R2 * (kron(inv(X' * X), Sigma)) * R2') * (R2 * B_vec - q2);
    pval2 = 1 - chi2cdf(wald_stat2, p);  % Calcul de la p-value

    % Résultats du test de Granger
    granger_results = table(...
        ["Y1 causes Y2"; "Y2 causes Y1"], ...
        [wald_stat1; wald_stat2], ...
        [pval1; pval2], ...
        ["No Cause"; "No Cause"], ...
        'VariableNames', {'Test', 'WaldStat', 'pValue', 'Decision'} ...
    );

    alpha = 0.05;
    granger_results.Decision(granger_results.pValue < alpha) = "Cause";  % Si p-value < alpha, on rejette l'hypothèse nulle
end
