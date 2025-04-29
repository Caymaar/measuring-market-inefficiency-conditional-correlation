function [cov_dcc, corr_dcc] = computeDCC(data1, data2, M, L, N, P, O, Q)
    % Compute DCC model for given data series and parameters
    %
    % Inputs:
    %   data1, data2 - Column vectors of data series
    %   M, L, N, P, O, Q - Parameters for the DCC model
    %
    % Outputs:
    %   cov_dcc - Covariance from the DCC model
    %   corr_dcc - Correlation from the DCC model

    % Combine the data series into a single matrix
    R = [data1, data2];

    % Estimation of scalar DCC
    [~, ~, Htdcc, ~] = dcc(R, [], M, L, N, P, O, Q);

    % Extract covariance and variances
    cov_dcc = squeeze(Htdcc(1, 2, :));
    v_1 = squeeze(Htdcc(1, 1, :));
    v_2 = squeeze(Htdcc(2, 2, :));

    % Compute correlation
    corr_dcc = cov_dcc ./ sqrt(v_1 .* v_2);
end
