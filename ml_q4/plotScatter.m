function output = plotScatter( data )
% Plot the sactter gragh
%   Need to add figure command before call this function


trainData = data(:,1:2);
trainLabels = data(:,3);
% trainData = hw4_nnet_train;
DataX = trainData(:,1);
DataY = trainData(:,2);
posDataX = DataX(trainLabels==1);
posDataY = DataY(trainLabels==1);
negDataX = DataX(trainLabels==-1);
negDataY = DataY(trainLabels==-1);
% plot the training data
scatter(posDataX,posDataY,'filled','r');
hold on
scatter(negDataX,negDataY,'b','*');

end

