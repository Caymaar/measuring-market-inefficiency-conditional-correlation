function [garchModels, condVols, residuals, bestPQ] = performGARCH(I_matrix, maxP, maxQ, useBIC)
% ESTIMATEGARCHPERSERIES - Estime un modèle GARCH optimal (p,q) pour chaque série.
%
% Entrées :
%   - I_matrix : matrice (T x N) des indices d'inefficience
%   - maxP     : ordre max de GARCH (p)
%   - maxQ     : ordre max de ARCH (q)
%   - useBIC   : booléen pour choisir BIC (true) ou AIC (false)
%
% Sorties :
%   - garchModels : cellule contenant les modèles GARCH estimés
%   - condVols    : volatilités conditionnelles (T x N)
%   - residuals   : résidus standardisés (T x N)
%   - bestPQ      : (N x 2) tableau des meilleurs (p,q)
%   - summaryTable: table récapitulative (Série, p, q)

    [T, N] = size(I_matrix);
    garchModels = cell(N, 1);
    condVols = NaN(T, N);
    residuals = NaN(T, N);
    bestPQ = zeros(N, 2);
    scores = NaN(N, 1);

    for i = 1:N
        serie = I_matrix(:, i);
        bestScore = Inf;
        bestModel = [];
        bestV = [];

        for p = 1:maxP
            for q = 1:maxQ
                try
                    mdl = garch(p, q);
                    [estModel, ~, logL] = estimate(mdl, serie, 'Display', 'off');
                    k = p + q + 1;

                    % Critère AIC/BIC
                    if useBIC
                        crit = log(T) * k - 2 * logL;
                    else
                        crit = 2 * k - 2 * logL;
                    end

                    if crit < bestScore
                        bestScore = crit;
                        bestModel = estModel;
                        [~, V] = infer(estModel, serie);
                        bestV = V;
                        bestPQ(i, :) = [p, q];
                        scores(i) = bestScore;
                    end
                catch
                    continue
                end
            end
        end

        if ~isempty(bestModel)
            garchModels{i} = bestModel;
            condVols(:, i) = sqrt(bestV);
            residuals(:, i) = serie ./ sqrt(bestV);
        else
            warning('Aucun modèle convergé pour la série %d', i);
        end
    end
end
