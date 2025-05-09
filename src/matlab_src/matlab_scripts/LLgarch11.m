function [LL,ll] = LLgarch11(theta, r)

    % Compute residuals
    e = r - theta(1);
    e = [0; e];  % This introduces a zero for the first value, might not be needed
    T = numel(r);  % Use original time series length
    v = zeros(T, 1);
    
    % Initial variance
    v0 = var(r);
    w = (1 - theta(2) - theta(3)) * v0;

    % Set initial variance
    v(1) = v0;  % Start with the initial variance

    for t = 2:T
        % GARCH(1,1) recursion
        v(t) = w + theta(2) * e(t-1)^2 + theta(3) * v(t-1);
    end
    
    % Log-likelihood calculation
    LL = T * 0.5 * log(2 * pi) + sum(log(v(2:end))) / 2 + 0.5 * sum((r(2:end) - theta(1)).^2 ./ v(2:end));
    ll =  T * 0.5 * log(2 * pi) + log(v(2:end)) / 2 + 0.5 * (r(2:end) - theta(1)).^2 ./ v(2:end);
end
