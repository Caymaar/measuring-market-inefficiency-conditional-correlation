function Archresults = performARCHTest(data, lags)
    varNames = data.Properties.VariableNames;
    n = numel(varNames);

    Archresults = table('Size', [n, 4], ...
        'VariableTypes', {'string', 'double', 'double', 'double'}, ...
        'VariableNames', {'MarketIndex', 'Chisquared', 'df', 'pValue'});

    for i = 1:n
        series = data.(varNames{i});
        [~, pValue, stat, ~] = archtest(series, 'Lags', lags);

        Archresults.MarketIndex(i) = varNames{i};
        Archresults.Chisquared(i) = stat;
        Archresults.df(i) = lags;
        Archresults.pValue(i) = pValue;
    end
end
