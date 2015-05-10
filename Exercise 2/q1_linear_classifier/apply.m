close all;
clear all;
clc;

% Load the data
train.data = load('lc_train_data.dat');
train.label = load('lc_train_label.dat');
test.data = load('lc_test_data.dat');
test.label = load('lc_test_label.dat');

% Train the classifier using the training dataset
[weight, bias] = leastSquares(train.data, train.label);

% Test the classifier on the training dataset
train.prediction = linclass(weight, bias, train.data);

% Print the performance of the classifier
train.acc = sum(train.prediction == train.label)/length(train.label);
fprintf('The accuracy of classifier on the training set is %g\n', train.acc);

% Define the range
xmax = max(train.data(:, 1));
xmin = min(train.data(:, 1));
ymax = max(train.data(:, 2));
ymin = min(train.data(:, 2));

train.figure = figure;

% Plot the data points and the decision line
hold on;
axis([xmin, xmax, ymin, ymax]);
axis equal;
title('Training Set');
plot(train.data(train.label==1, 1), train.data(train.label==1, 2), 'bx')
plot(train.data(train.label==-1, 1), train.data(train.label==-1, 2), 'ro');
line([xmin, xmax], [-(weight(1)*xmin+bias)/weight(2), -(weight(1)*xmax+bias)/weight(2)]);

% Test the classifier on the test dataset
test.prediction = linclass(weight, bias, test.data);

% Print the performance of the classifier
test.acc = sum(test.prediction==test.label)/length(test.label);
fprintf('The accuracy of classifier on the test set is %g\n', test.acc);

% Define the range
xmax = max(test.data(:, 1));
xmin = min(test.data(:, 1));
ymax = max(test.data(:, 2));
ymin = min(test.data(:, 2));

% Plot the data points and the decision line
test.figure = figure;
hold on;
axis([xmin, xmax, ymin, ymax]);
axis equal;
title('Test Set');
plot(test.data(test.label==1, 1), test.data(test.label==1, 2), 'bx');
plot(test.data(test.label==-1, 1), test.data(test.label==-1, 2), 'ro');
line([xmin, xmax], [-(weight(1)*xmin+bias)/weight(2), -(weight(1)*xmax+bias)/weight(2)]);

