function ADFresults = performADFTest(data, lags)
    varNames = data.Properties.VariableNames;
    n = numel(varNames);

    ADFresults = table('Size', [n, 4], ...
        'VariableTypes', {'string', 'double', 'double', 'double'}, ...
        'VariableNames', {'MarketIndex', 'ADFStatistic', 'LagOrder', 'pValue'});

    for i = 1:n
        series = data.(varNames{i});
        [~, p, stat, ~, ~] = adftest(series, 'Lags', lags);

        ADFresults.MarketIndex(i) = varNames{i};
        ADFresults.ADFStatistic(i) = stat;
        ADFresults.LagOrder(i) = lags;
        ADFresults.pValue(i) = p;
    end
end
