function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

m = size(X,1); % Number of examples
for i = 1:m
    x = X(i,:); % current data point
    c = centroids(1,:); % centroid data point
    index = 1; % index of current centroid
    mat = [x;c];
    
    % Compute the first squared distance and set it to the minimum
    dist = pdist(mat,'euclidean')*pdist(mat,'euclidean');
    min_dist = dist;
    
    % Loop through other centroids
    for j = 2:K
        c_new = centroids(j,:);
        mat = [x;c_new];
        dist = pdist(mat,'euclidean')*pdist(mat,'euclidean');
        % If we have a smaller squared distance, update the index
        if (dist < min_dist)
            min_dist = dist;
            index = j;
        end
    end
    
    % Set the Centroid index
    idx(i) = index;
end






% =============================================================

end

