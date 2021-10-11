% A test script for Assignment 1
% Written by Asher Hancock
% October 10, 2021

close all
clear all

%{
% Read in the Data
data = load("ex1data1.txt");
m = size(data,1); % Number of Training Examples
n = size(data,2)-1; % Number of Features

% Create the Data Input Array where the first column is 1s
% This is due to the theta0 multiplication
X = ones(m, 1);
X = [X data(:, 1)];

% Create the Data Output Array
y = data(:, 2);

% Create a theta array
theta = ones(n+1, 1);

% Compute the Cost
J = computeCost(X, y, theta);

% Compute Gradient Descent
alpha = 0.0005;
num_iters = 1000;
[theta, J_hist] = gradientDescent(X, y, theta, alpha, num_iters);

% Normalize
[X_norm, mu, sigma] = featureNormalize(X);

% Plot data
figure(1)
plot(X(:,2), y, 'ro');
hold on

new_y = theta(1,1) + theta(2,1)*X(:,2);
plot(X(:,2), new_y, 'b-');

theta_check = normalEqn(X,y);
new_y2 = theta_check(1,1) + theta_check(2,1)*X(:,2);
plot(X(:,2), new_y2, 'g-');
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%{
% New Data
data2 = load("ex1data2.txt");
m = size(data2,1); % Number of Training Examples
n = size(data2,2)-1; % Number of Features

% Create the Data Input Array where the first column is 1s
% This is due to the theta0 multiplication
X = ones(m, 1);
X = [X data2(:, 1:n)];

% Create the Data Output Array
y = data2(:, n+1);

% Create a theta array
theta = ones(n+1, 1);

% Compute the Cost
J = computeCostMulti(X, y, theta);

alpha = 0.0001;
num_iters = 10000;
% Gradient Descent
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Normal Equation
theta_check = normalEqn(X,y);
%}
