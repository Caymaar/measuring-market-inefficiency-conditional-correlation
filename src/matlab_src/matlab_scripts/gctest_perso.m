function varargout = gctest_perso(Data,varargin)
%GCTEST Granger causality and block exogeneity tests
%
% Syntax:
%
%   [h,pValue,stat,cValue] = gctest(Y1,Y2)
%   [...] = gctest(Y1,Y2,Y3)
%   [...] = gctest(Y1,Y2,param,val,...)
%   [...] = gctest(Y1,Y2,Y3,param,val,...)
%   StatTbl = gctest(Tbl)
%   StatTbl = gctest(Tbl,param,val,...)
%
% Description:
%
%   Granger causality and block exogeneity tests assess whether past values
%   of the variable Y1 have an impact on the predictive distribution of the
%   variable Y2, if values of the variable Y3 are given. Tests are
%   conducted in the vector autoregression (VAR) framework. The null
%   hypothesis is that Y1 is not a one-step Granger-cause of Y2.
%
% Input Arguments:
%
%   Y1 - Univariate or multivariate series representing "Causes",
%        specified as a column vector or matrix.
%
%   Y2 - Univariate or multivariate series representing "Effects",
%        specified as a column vector or matrix.
%
%   Y3 - Univariate or multivariate series for conditioning, specified as a
%        column vector or matrix. Y3 is optional, and the
%        default value is empty, indicating no conditioning variables.
%
%   Tbl - Multivariate series containing all variables in the test,
%        specified as a table or timetable. The 'CauseVariables' parameter
%        specifies the causes, the 'EffectVariables' parameter specifies
%        the effects, the 'ConditionVariables' parameter specifies the
%        conditioning series, and the 'PredictorVariables' parameter
%        specifies the exogenous predictor variables in the VAR model.
%
%   The function removes NaN values indicating missing observations.
%
% Optional Input Parameter Name/Value Arguments:
%
% 'NumLags'     Nonnegative integer scalar number of lagged responses (p) 
%               included in the underlying VAR(p) model. The default is 1.
%
% 'Integration' Nonnegative integer scalar maximum order of integration of 
%               data series. The default is 0 (Data are stationary).
%
% 'Constant'    Logical scalar indicator to include a constant. The default
%               is true.
%
% 'Trend'       Logical scalar indicator to include a linear time trend. 
%               The default is false.
%
% 'X'           Predictor data corresponding to a regression component in
%               the VAR model, specified as a matrix. See VARM.ESTIMATE. 
%               The default is an empty series and the estimated VAR model
%               has no exogenous predictors.
% 
% 'Alpha'       Positive value of significance level for tests. The default
%               is 0.05. 
%
% 'Test'        String or character vector indicating test statistics. 
%               Values are 'chi-square' or 'f'. The default is 'chi-square'.
%
% 'CauseVariables' Variables in Tbl to use for the causes,
%               specified as names in Tbl.Properties.VariableNames.
%               Variable names are cell vectors of character vectors,
%               string vectors, integer vectors or logical vectors. 
%               The default is all variables in Tbl except the value of
%               'EffectVariables'.
%
% 'EffectVariables' Variables in Tbl to use for the effects,
%               specified as names in Tbl.Properties.VariableNames.
%               Variable names are cell vectors of character vectors,
%               string vectors, integer vectors or logical vectors. 
%               The default is the last variable in Tbl.
%
% 'ConditionVariables' Variables in Tbl to use for the conditioning series,
%               specified as names in Tbl.Properties.VariableNames.
%               Variable names are cell vectors of character vectors,
%               string vectors, integer vectors or logical vectors. 
%               The default is empty.
%
% 'PredictorVariables' Variables in Tbl to use for the exogenous predictors 
%               in the VAR model (see also 'X'), specified as names in
%               Tbl.Properties.VariableNames. Variable names are cell
%               vectors of character vectors, string vectors, integer
%               vectors or logical vectors. The default is empty.
%
% Output Arguments:
%
% h             Logical value of decisions for the test. Values of h equal 
%               to 1 indicate rejection of the null hypothesis in favor of 
%               Granger causality and endogeneity. Values of h equal to 0 
%               indicate failure to reject the null hypothesis of one-step 
%               non-causality.
%
% pValue        p-value of the test statistic.
%
% stat          Test statistic.
%
% cValue        Critical value of the test.
%
% StatTbl       When input is Tbl, outputs h, pValue, stat, cValue, alpha 
%               and test are returned in the table StatTbl, with a row for 
%               each test.
%
% Notes:
%
%   o The number of observations (rows) in [Y1 Y2 Y3] relates to the sample
%     size available for estimation. If the number of observations in Y1,
%     Y2, and Y3 differ, then only the most recent observations common
%     to all are retained.
%
%   o Variables in Y1, Y2, Y3 and X must be distinct.
%
%   o Integrated series are handled by level VAR with augmented lags.
%
% References:
%
%   [1] Granger, C. W. J. Investigating Causal Relations by Econometric
%       Models and Cross-spectral Methods, Econometrica, 1969, 37, 424-459.
%
%   [2] Hamilton, J. D. Time Series Analysis. Princeton, NJ: Princeton
%       University Press, 1994.
%
%   [3] Lutkepohl, H. New Introduction to Multiple Time Series Analysis.
%       Springer-Verlag, 2007.
%
%   [4] Toda, H. Y and T. Yamamoto. Statistical inferences in vector 
%       autoregressions with possibly integrated processes. Journal of 
%       Econometrics, 1995, 66, 225-250.
%
%   [5] Dolado, J. J. and H. Lutkepohl. Making Wald Tests Work for  
%       Cointegrated VAR Systems. Econometric Reviews, 1996, 15, 369-386.

% Copyright 2023 The MathWorks, Inc.

isTabular = istable(Data) || istimetable(Data);

% Parse inputs and set defaults
callerName = 'gctest';
parseObj = inputParser;
parseObj.addRequired('Data',@(x)validateattributes(x,{'double','table','timetable'},{'nonempty','2d'},callerName));
if ~isTabular
    parseObj.addRequired('Y2',@(x)validateattributes(x,{'double'},{'2d'},callerName));
    parseObj.addOptional('Y3',[],@(x)validateattributes(x,{'double'},{'2d'},callerName));
end
parseObj.addParameter('NumLags',1,@(x)validateattributes(x,{'double','logical'},{'scalar','nonnegative','integer'},callerName));
parseObj.addParameter('Integration',0,@(x)validateattributes(x,{'double','logical'},{'scalar','nonnegative','integer'},callerName));
parseObj.addParameter('Constant',true,@(x)validateattributes(x,{'double','logical'},{'scalar','binary'},callerName));
parseObj.addParameter('Trend',false,@(x)validateattributes(x,{'double','logical'},{'scalar','binary'},callerName));
parseObj.addParameter('X',[],@(x)validateattributes(x,{'double','logical','char','string','cell'},{},callerName));
parseObj.addParameter('Alpha',0.05,@(x)validateattributes(x,{'numeric'},{'positive','<',1},callerName));
parseObj.addParameter('Test','chi-square',@(x)validateattributes(x,{'char','string','cell'},{},callerName));
parseObj.addParameter('CauseVariables',[],@(x)validateattributes(x,{'double','logical','char','string','cell'},{'vector'},callerName));
parseObj.addParameter('EffectVariables',[],@(x)validateattributes(x,{'double','logical','char','string','cell'},{'vector'},callerName));
parseObj.addParameter('ConditionVariables',[],@(x)validateattributes(x,{'double','logical','char','string','cell'},{'vector'},callerName));
parseObj.addParameter('PredictorVariables',[],@(x)validateattributes(x,{'double','logical','char','string','cell'},{'vector'},callerName));
parseObj.parse(Data,varargin{:});
% Data = parseObj.Results.Data;
nlag = parseObj.Results.NumLags;
integration = parseObj.Results.Integration;
Constant = parseObj.Results.Constant;
Trend = parseObj.Results.Trend;
X = parseObj.Results.X;
alpha = parseObj.Results.Alpha;
chi2Str = parseObj.Results.Test;
CauseVariables = parseObj.Results.CauseVariables;
EffectVariables = parseObj.Results.EffectVariables;
ConditionVariables = parseObj.Results.ConditionVariables;
PredictorVariables = parseObj.Results.PredictorVariables;

% Test statistics are chi-square or f
chi2Str = string(chi2Str);
chi2Flag = strncmpi(chi2Str,'chi-square',1);
fFlag = strncmpi(chi2Str,'f',1);
if ~all(chi2Flag | fFlag)
    error(message('econ:gctest:Chi2F'))
end

% Extract Y1, Y2, Y3 and X from Data
if isTabular
    
    % Default effect variable is the last column of the table
    if isempty(EffectVariables)
        EffectVariables = size(Data,2);
    end
    Y2 = Data(:,EffectVariables);
    
    % Default cause variables are all columns other than effect variables
    if isempty(CauseVariables)
        Y1 = Data;
        Y1(:,EffectVariables) = [];
    else
        Y1 = Data(:,CauseVariables);
    end
    
    % Default conditioning variable is empty
    Y3 = Data(:,ConditionVariables);
    
    % Exogenous predictor variables, specified by X or PredictorVariables
    if ~isempty(PredictorVariables)
        % Use PredictorVariables for variable selection
        if ~isempty(X)
            error(message('econ:gctest:SelectX'))
        end
        X = table2array(Data(:,PredictorVariables));
    elseif ~isempty(X) && isvector(X) && numel(X)<=size(Data,2) && numel(X)~=size(Data,1)
        % Use X for variable selection
        X = table2array(Data(:,X));
    end
        
    % Validate table inputs
    internal.econ.TableAndTimeTableUtilities.isTabularFormatValid(Y1,'Y1');
    internal.econ.TableAndTimeTableUtilities.isTabularFormatValid(Y2,'Y2');
    internal.econ.TableAndTimeTableUtilities.isTabularFormatValid(Y3,'Y3');
    if ~isempty(Y1)
        internal.econ.TableAndTimeTableUtilities.isTabularDataSinglePath(Y1,'Y1');
    end
    if ~isempty(Y2)
        internal.econ.TableAndTimeTableUtilities.isTabularDataSinglePath(Y2,'Y2');
    end
    if ~isempty(Y3)
        internal.econ.TableAndTimeTableUtilities.isTabularDataSinglePath(Y3,'Y3');
    end
    
    % Variables in Y1, Y2 and Y3 must be distinct
    Y1Names = Y1.Properties.VariableNames;
    Y2Names = Y2.Properties.VariableNames;
    Y3Names = Y3.Properties.VariableNames;
    if any(ismember(Y2Names,Y1Names)) || any(ismember(Y3Names,Y1Names)) || any(ismember(Y3Names,Y2Names))
        error(message('econ:gctest:DistinctVariables'))
    end
    
    % Convert tables to numeric values
    Y1 = table2array(Y1);
    Y2 = table2array(Y2);
    Y3 = table2array(Y3);
else
    Y1 = Data;
    Y2 = parseObj.Results.Y2;
    Y3 = parseObj.Results.Y3;
    
    if ~isempty(CauseVariables)
        warning(message('econ:gctest:UnusedCause'))
    end
    if ~isempty(EffectVariables)
        warning(message('econ:gctest:UnusedEffect'))
    end
    if ~isempty(ConditionVariables)
        warning(message('econ:gctest:UnusedCondition'))
    end
end
validateattributes(X,{'numeric'},{'2d'},callerName,'X');

% VAR data
nobsY1 = size(Y1,1);
nobsY2 = size(Y2,1);
nobsY3 = size(Y3,1);
nobsX = size(X,1);
if nobsY3 == 0
    nobsY3 = min([nobsY1,nobsY2]);
    Y3 = zeros(nobsY3,0,'like',Y1);
end
if nobsX == 0
    nobsX = min([nobsY1,nobsY2,nobsY3]);
    X = zeros(nobsX,0,'like',Y1);
end
nobs = min([nobsY1,nobsY2,nobsY3,nobsX]);
if nobsY1 > nobs
    Y1 = Y1(nobsY1-nobs+1:end,:);
end
if nobsY2 > nobs
    Y2 = Y2(nobsY2-nobs+1:end,:);
end
if nobsY3 > nobs
    Y3 = Y3(nobsY3-nobs+1:end,:);
end
if nobsX > nobs
    X = X(nobsX-nobs+1:end,:);
end
Y123 = [Y1,Y2,Y3];

% VAR dimension
nY1 = size(Y1,2);
nY2 = size(Y2,2);
nY3 = size(Y3,2);
nvar = nY1 + nY2 + nY3;
nX = size(X,2);
numPredictors = nvar*nlag + Constant + Trend + nX;

% Actual sample size
missingValue = any(isnan([Y123,X]),2);
nobs = nobs - sum(missingValue) - (nlag+integration);

% Variables in VAR must be distinct
Y123cut = Y123(~missingValue,:);
Xcut = X(~missingValue,:);
for m = 1:nX
    if any(all(Xcut(:,m)==Y123cut,1))
        error(message('econ:gctest:DistinctVariables'))
    end
end

for m = 1:nvar
    if sum(all(Y123cut(:,m)==Y123cut,1)) > 1
        error(message('econ:gctest:DistinctVariables'))
    end
end

% Ensure real, finite data:

D = [Y123cut,Xcut];

if any(~isreal(D(:))) || any(~isfinite(D(:)))

    error(message('econ:gctest:DataError'))

end

% Construct n-dimensional VAR(p) augmented with additional lags
Mdl = varm(nvar,nlag+integration);
if ~Constant
    Mdl.Constant(:) = 0;
end
if Trend
    Mdl.Trend(:) = NaN;
end

% Estimate VAR model
% Covariance of estimated VAR coefficients will be used by Granger tests
if isempty(X)
    [EstMdl,~,~,~,Sigma] = estimate(Mdl,Y123);
else
    [EstMdl,~,~,~,Sigma] = estimate(Mdl,Y123,'X',X);
end
CoeffAR = [EstMdl.AR{:}];
finiteSampleAdjust = nobs ./ (nobs - numPredictors);
CovAR = Sigma.AR .* finiteSampleAdjust;

% Subset of causes and effects
Cause = 1:nY1;
Effect = nY1+1:nY1+nY2;

% Extract sub-vector and sub-matrix
IndMat = reshape(1:nvar*nvar*nlag,[nvar,nvar,nlag]);
ind = zeros(0,1);
for m = 1:nlag
    submatrix = IndMat(Effect,Cause,m);
    ind = [ind;submatrix(:)]; %#ok<AGROW>
end
CovARcut = CovAR(ind,ind);
CoeffARcut = CoeffAR(ind);

% Test statistics under the null hypothesis
% For F(df1,df2) test, assume that df2=nobs-numPredictors
% It is also reasonable to use df2=(nobs-numPredictors)*nvar
df1 = numel(CoeffARcut);
df2 = nobs - numPredictors;

% If degree of freedom is zero like VAR(0), chi2cdf and fcdf may not work
% In that case, Granger non-causality holds trivially
% h=0; stat=0; pValue=1; cValue=0;
% Alternatively, put a small value and apply the usual inference procedure
if df1 == 0
    df1 = 0.001;
end

% Prepare for statistics
[CovARChol,flag] = chol(CovARcut,'lower');
if flag > 0
    CovARcut = CovARcut + 1e-8 * eye(size(CovARcut),'like',CovARcut);
    CovARChol = chol(CovARcut,'lower');
end
normalRV = CovARChol \ CoeffARcut;
normalRV2 = normalRV.' * normalRV;

% Tests under multiple significance levels or different test statistics
numResults = max(numel(chi2Flag),numel(alpha));
if numResults > 1
    if isscalar(chi2Flag)
        chi2Flag = repmat(chi2Flag,size(alpha));
    elseif isscalar(alpha)
        alpha = repmat(alpha,size(chi2Flag));
    else
        validateattributes(alpha,{'numeric'},{'size',size(chi2Flag)},'','Alpha');
    end        
end
dim = size(alpha);
stat = zeros(dim,'like',normalRV2);
pValue = zeros(dim,'like',normalRV2);
cValue = zeros(dim,'like',normalRV2);
testString = strings(dim);

% Test statistics
for m = 1:numResults
    if chi2Flag(m)
        stat(m) = normalRV2;
        pValue(m) = 1 - chi2cdf(stat(m), df1);
        cValue(m) = chi2inv(1-alpha(m),df1);
        testString(m) = "chi-square";
    else
        stat(m) = normalRV2 / df1;
        pValue(m) = fcdf(stat(m),df1,df2,'upper');
        cValue(m) = finv(1-alpha(m),df1,df2);
        testString(m) = "f";
    end    
end
h = pValue <= alpha;

% Create output table
if isTabular
    RowNames = "Test " + (1:numResults);
    VariableNames = ["h", "pValue", "stat", "cValue","alpha","test"];
    StatTbl = table(h(:),pValue(:),stat(:),cValue(:),alpha(:),testString(:),'RowNames',RowNames,'VariableNames',VariableNames);
end

% Assign outputs to varargout
if isTabular
    nargoutchk(0,1);
    varargout{1} = StatTbl;
else
    nargoutchk(0,4);
    switch nargout
        case {0,1}
            varargout{1} = h;
        case 2
            varargout{1} = h;
            varargout{2} = pValue;            
        case 3            
            varargout{1} = h;
            varargout{2} = pValue;
            varargout{3} = stat;            
        case 4            
            varargout{1} = h;
            varargout{2} = pValue;
            varargout{3} = stat;
            varargout{4} = cValue;
    end
end
