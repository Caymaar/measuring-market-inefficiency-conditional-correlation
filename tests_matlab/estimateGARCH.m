function [condVols, residuals, aics, pqs] = estimateGARCH(data, maxP, maxQ)
    if istable(data)
        seriesNames = data.Properties.VariableNames;
        data = table2array(data);
    else
        [~, N] = size(data);
        seriesNames = strcat("Serie_", string(1:N));
    end

    [T, N] = size(data);
    condVols = NaN(T, N);
    residuals = NaN(T, N);
    aics = NaN(1, N);
    pqs = zeros(N, 2);

    for i = 1:N
        r = data(:, i);
        try
            [model, pq, aic, varcond, zt] = fitBestGARCH(r, maxP, maxQ);
            condVols(:, i) = sqrt(varcond);
            residuals(:, i) = zt;
            aics(i) = aic;
            pqs(i, :) = pq;
        catch
            warning("Erreur sur la série %d", i);
        end
    end
end

function [bestModel, bestPQ, bestAIC, varcond, zt] = fitBestGARCH(r, maxP, maxQ)
    bestAIC = inf;
    bestModel = [];
    bestPQ = [0, 0];

    for p = 0:maxP
        for q = 0:maxQ
            try
                model = garch(p, q);
                [estModel, ~, logL] = estimate(model, r, 'Display', 'off');

                numParams = p + q + 1; % omega + alphas + betas
                aic = 2 * numParams - 2 * logL;

                if aic < bestAIC
                    bestAIC = aic;
                    bestModel = estModel;
                    bestPQ = [p, q];
                end
            catch
                continue;
            end
        end
    end

    if ~isempty(bestModel)
        [varcond, ~] = infer(bestModel, r);
        mu_hat = mean(r); 
        zt = (r - mu_hat) ./ sqrt(varcond);
    else
        varcond = NaN(size(r));
        zt = NaN(size(r));
    end
end

