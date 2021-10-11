function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
n = size(theta,1);
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    
    % Iterative Version
    theta_temp = theta;
    for j = 1:n
        sum = 0;
        for i = 1:m
            x = X(i,:);
            sum = sum + ((theta'*x' - y(i)))*x(j);
        end
        theta_temp(j) = theta(j) - alpha*(1/m)*sum;
    end
    theta = theta_temp;
    J_history(iter) = computeCostMulti(X, y, theta);
    
    % Vectorized Version
    %{
    theta_temp = theta - alpha * (1/m) * (((X*theta) - y)' * X)';
    theta = theta_temp;
    J_history(iter) = computeCostMulti(X, y, theta);
    %}
end

end
