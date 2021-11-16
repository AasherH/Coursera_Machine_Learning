function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% Compute the Cost Function
% Below, both yield the same numerical answer. However, only with the
% second cost function was I able to get the submit.m function to work
J =  (1/(2*m))*sum(((X*theta - y).^2)) + (lambda/(2*m))*sum(((theta(2,end)).^2));
J = (1/(2*m))*(X*theta-y)'*(X*theta-y) + (lambda/(2*m))*(theta(2:length(theta)))'*theta(2:length(theta));

% Compute the Gradients
partial1 = (1/m)*(X*theta-y)'*X;
partial2 = (lambda/m)*theta';
partial2(1) = 0; % Don't regularize the theta0 term

grad = (partial1+partial2)';

% =========================================================================

grad = grad(:);

end
