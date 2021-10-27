function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

%%
% ====================== YOUR CODE HERE ======================
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. 

% Compute the Feedforward Portion of the NN
% Add ones to the X data matrix
X = [ones(m, 1) X]; % 5000x401

% Compute the Input -> Hidden Layer
z1 = X*Theta1'; % 5000x25
a1 = sigmoid(z1); % 5000x25
    
% Add ones to a1 for bias
a1 = [ones(m,1), a1]; % 5000x26

% Compute the Hidden Layer -> Output Layer (Predictions)
z2 = a1*Theta2'; % 5000x10
a2 = sigmoid(z2); % 5000x10

% Loop through each class to create a cost function
for k = 1:num_labels
    % Create a boolean vector for the number of labels
    y_temp = (y == k); % 5000x1 logical
    
    % Extract the class prediction from the a2 vector and use Logistic
    % Regression
    j1 = -y_temp'*log(a2(:,k)); % 1x5000 x 5000x1 = 1x1
    j2 = ((1-y_temp)'*log(1-a2(:,k))); % 1x5000 x 5000x1 = 1x1
    J = J + (j1-j2); % 1x1
end
% Average the Cost Functions
J = (1/m)*J;

% Now, add the Regularization Terms

% Do not regularize the bias thetas. This corresponds to the first column
Theta1_Temp = Theta1;
Theta1_Temp(:,1) = 0;

Theta2_Temp = Theta2;
Theta2_Temp(:,1) = 0;

% Note the double sum, which sums the rows and cols
reg_sum = (lambda/(2*m))*(sum(sum(Theta1_Temp.^2)) + sum(sum(Theta2_Temp.^2)));

% Re-compute the cost function
J = J + reg_sum;

%%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

big_delta2 = zeros(size(Theta2))';
big_delta1 = zeros(size(Theta1))';

y_new = zeros(num_labels, m);
for i = 1:m
    y_new(y(i),i)=1;
end

% For every training example, we need to perform backprop
for t = 1:m
    % Perform a feedforward pass for each training example
    a1 = X(t,:); % 1x401
    z2 = a1*Theta1'; % 1x25
    a2 = sigmoid(z2); % 1x25
    a2 = [ones(1,1), a2]; % 1x26
    z3 = a2*Theta2'; % 1x10
    a3 = sigmoid(z3); % 1x10
    
    % After the feedforward pass, we now can perform backprop
    % Compute the Error Terms
    a3 = a3'; % 10x1
    little_delta3 = a3 - y_new(:,t); % 10x1
    
    % Account for Bias in z2
    % This is needed to match the dimensions of Theta2
    z2 = [ones(1,1) z2]'; % 26x1
    
    
    little_delta2 = (Theta2'*little_delta3).*sigmoidGradient(z2); % 26x1
    
    % Skip the Bias Term when performing a derivative
    little_delta2 = little_delta2(2:end); % 25x1
 
    % Compute the gradients
    % big_delta# match the notes from this section
    Theta2_grad = Theta2_grad + (little_delta3 * a2); % (10x1)x(1x26)
    %big_delta2 = big_delta2 + (little_delta3*a2)';
	Theta1_grad = Theta1_grad + (little_delta2 * a1); % (25x1)x(1x401)
    %big_delta1 = big_delta1 + (little_delta2*a1)';
end

%Theta2_grad = (1/m)*big_delta2';
%Theta1_grad = (1/m)*big_delta1';
Theta2_grad = (1/m)*Theta2_grad;
Theta1_grad = (1/m)*Theta1_grad;
%%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
reg_theta1 = (lambda/m)*Theta1(:, 2:end);
reg_theta2 = (lambda/m)*Theta2(:, 2:end);

Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + reg_theta1;
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + reg_theta2;



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
