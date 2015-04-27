function pred = predictNN( w1,w2,X )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

m = size(X,2); % number of samples
X = [ones(1,m);X]; % add a bia unit

% step 2 : forward
score1 = w1'*X; % 6x1
X1 = [ones(1,m);tanh(score1)];
score2 = w2'*X1;
X2 = tanh(score2);
[dummy, pred] = max(X2, [], 2);

end

