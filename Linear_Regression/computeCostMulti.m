function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % Number of training examples
n = size(X,2); % Number of features

% Initialize the Cost
J = 0;

% Iterative Solution
for i = 1:m
    sum1 = 0;
    for j = 1:n
        sum1 = sum1+ theta(j,1)*X(i,j);
    end
    J = J + (sum1-y(i))^2;
end
J = J*1/(2*m);

end
