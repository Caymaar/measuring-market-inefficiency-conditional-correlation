function [ARCHStats, ARCHpValues] = performARCHTest(data, lags)
    varNames = data.Properties.VariableNames;
    n = numel(varNames);

    ARCHStats = zeros(1, n);
    ARCHpValues = zeros(1, n);

    for i = 1:n
        series = rmmissing(data.(varNames{i}));
        [~, pValue, stat] = archtest(series, 'Lags', lags);
        ARCHStats(i) = stat;
        ARCHpValues(i) = pValue;
    end
end
