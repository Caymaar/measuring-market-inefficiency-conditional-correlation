function h2 = granger_test(data1, data2)
    % Convertir en tableau si table
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
    data1 = double(data1);  % Force vecteur colonne
    data2 = double(data2);
    % Test: est-ce que data1 cause data2 ?
    h1 = gctest(data1, data2, NumLags=1);

    % Test: est-ce que data2 cause data1 ?
    h2 = gctest(data2, data1, NumLags=1);

    % Résultats
    %results = table([h1; h2], [p1; p2], [stat1; stat2], [cValue1; cValue2], ...
      %  'VariableNames', {'Reject', 'pValue', 'FStat', 'CritValue'}, ...
     %   'RowNames', {'data1 => data2', 'data2 => data1'});
 
end
