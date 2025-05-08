function [ADFStats, ADFpValues] = performADFTest(data, lags)
    varNames = data.Properties.VariableNames;
    n = numel(varNames);

    ADFStats = zeros(1, n);
    ADFpValues = zeros(1, n);

    for i = 1:n
        series = data.(varNames{i});  % ✅ Correct
        [~, p, stat] = adftest(series, 'Lags', lags);
        ADFStats(i) = stat;
        ADFpValues(i) = p;
    end
end
