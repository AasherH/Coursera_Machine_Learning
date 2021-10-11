function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
n = size(X,2);
theta0 = theta(1,1);
theta1 = theta(2,1);
for iter = 1:num_iters
    % Iterative/Hard-Coded Solution
    for j = 1:n
        temp = 0;
        for i = 1:m
            if (j == 1)
                temp = temp + (theta0 + theta1*X(i,2) - y(i));
            end
            if (j > 1)
                temp = temp + (theta0 + theta1*X(i,2) - y(i))*X(i,2);
            end
        end
        theta(j,1) = theta(j,1) - alpha*(1/m)*temp;
    end
    theta0 = theta(1,1);
    theta1 = theta(2,1);
    J_history(iter,1) = computeCost(X,y,theta);
end

end
