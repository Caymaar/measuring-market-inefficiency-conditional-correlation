function [condVols, residuals] = estimateGARCH(data)
    % data : table ou matrice (T x N) de séries temporelles

    % Si data est une table, convertis-la en matrice et garde les noms
    
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

    % Boucle sur chaque série
    for i = 1:N
        r = data(:, i);
        try
            [mu, alpha, beta, varcond, zt] = fitGARCH(r);
            condVols(:, i) = sqrt(varcond);
            residuals(:, i) = zt;
        catch
            warning('Erreur sur la série %d', i);
        end
    end
end

function [mu_hat, alpha_hat, beta_hat, varcond, zt] = fitGARCH(r)
    mu = mean(r);
    alpha = 0.15;
    beta = 0.8;
    theta0 = [mu, alpha, beta];

    ll = @(x) LLgarch11(x, r);
    options = optimoptions('fminunc','MaxIterations',500,'Display','off');
    [x, ~, exitflag] = fminunc(ll, theta0, options);

    if exitflag <= 0
        warning('Optimisation non convergée');
    end

    T = length(r);
    e = (r - x(1)).^2;
    varcond = zeros(T,1);
    varcond(1) = var(r);
    w = (1 - x(2) - x(3)) * var(r);

    for t = 2:T
        varcond(t) = w + x(2) * e(t-1) + x(3) * varcond(t-1);
    end

    zt = (r - x(1)) ./ sqrt(varcond);

    mu_hat = x(1);
    alpha_hat = x(2);
    beta_hat = x(3);
end
