%% Machine Learning Technique Homework 4

%% Experienment with Backprop Neural Network 
%
%
%% Initialization
clear ; close all; clc

%% loading dataset
% training set - hw4_nnet_train : 25x3, each row (xn1, xn2, yn)
load('hw4_nnet_train.dat');
% test set - hw4_nnet_train: 250x3,
load('hw4_nnet_test.dat');

% each sample with one column
% trainData : 2x25
trainDataset = hw4_nnet_train;
trainData = hw4_nnet_train(:,1:2)';
trainLabels = hw4_nnet_train(:,3)';

testDataset = hw4_nnet_test;
testData = hw4_nnet_test(:,1:2)';
testLabels = hw4_nnet_test(:,3)';

% DataX = trainData(:,1);
% DataY = trainData(:,2);
% posDataX = DataX(trainLabels==1);
% posDataY = DataY(trainLabels==1);
% negDataX = DataX(trainLabels==-1);
% negDataY = DataY(trainLabels==-1);
% plot the training data
% figure(1);
% plotScatter(trainDataset);
% 
% % test set
% figure(2);
% plotScatter(testDataset);

%% Backprop Neural Network 

% don't work = =, Todo
% set parameters
inputSize = size(trainData, 1); % 2
numSample = size(trainData, 2); % 25
hiddenSizeList = [1, 6, 11, 16, 21];
hiddenSize = 21;
outputSize = 1;
eta = 0.1;
range = 0.1;
itrnum = 50000;

% random initilize the weight
% bia unit?
% w1: 3x6, w2:7x1
w1 = [ones(1,hiddenSize);2*range*(rand(inputSize, hiddenSize)-0.5)];
w2 = [ones(1,outputSize);2*range*(rand(hiddenSize, outputSize)-0.5)];
score1 = zeros(hiddenSize,numSample);
X1 = [ones(1,numSample);zeros(hiddenSize,numSample)];
score2 = zeros(outputSize,numSample);
X2 = [zeros(outputSize,numSample)];
delta2 = zeros(outputSize,numSample);
delta1 = zeros(hiddenSize,numSample);

% Backpropagation 
for i = 1:itrnum
% step 1 : stochastic
% randomly pick up one sample
numcol = randi(numSample); % numcol:j
Y = trainLabels(:,numcol);
X0 = trainData(:,numcol); % column vector : 2x1
X0 = [1;X0]; % add a bia unit

% step 2 : forward
score1 = w1'*X0; % 6x1
X1(:,numcol) = [1;tanh(score1)];
score2 = w2'*X1;
X2 = tanh(score2);

% step 3 : backward
% check?
grad2 = tanhgradient(score2);
delta2(:,numcol) = -2*(Y - X2(:,numcol)).*(grad2(:,numcol)); % d_L+1 x 1
delta1(:,numcol) = w2(2:size(w2,1),:) * delta2(:,numcol)' .*(tanhgradient(score1)) ;  % d_l+1 x 1

% step 4 : gradient descent
% how to update the weight of bias unit?
w1(2:size(w1,1),:) = w1(2:size(w1,1),:) - eta*trainData*delta1';
w2 = w2 - eta*X1*delta2';

end

% return hypothesis?
% prediction

pred = predictNN(w1, w2, trainData);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == trainLabels)) * 100);



