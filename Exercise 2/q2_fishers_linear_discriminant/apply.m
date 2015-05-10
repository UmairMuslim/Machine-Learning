close all;
clear all;
clc;

p = parameters();

disp('creating datasets');
% Dataset X1
mean_X1 = [1, 1]';
sigma_X1 = [10, 1; 1, 1];

% Dataset X2
mean_X2 = [-12, 6]';
sigma_X2 = [3, 1; 1, 1];

% Create the two datasets
X1 = gsample(mean_X1, sigma_X1, p.sample_size(1))';
X2 = gsample(mean_X2, sigma_X2, p.sample_size(2))';
disp('datasets created');

% Compute the weight vector
w = fisherLDA(X1, X2);

% Project both datasets X1 and X2 on the line which is defined by w
[Y1, Y2] = project(w, X1, X2);

% Estimate w0
w0 = estimateW0(Y1, Y2);

% Plot the datasets
figure;
hold on;
plot(X1(:,1), X1(:,2), 'rx')
plot(X2(:,1), X2(:,2), 'g*');
axis([-20 20 -20 20]);
axis equal;

% Plot the lines. Blue line is the line where we project the
% points and the green line is the decision boundary that we use
% for classifying.
x = linspace(-20, 20);
g = (w(2) .* x) / w(1);
y = -(w(1) .* x) / w(2) + w0 / w(2);
plot(x, g, 'b-');
plot(x, y, 'g-');
hold off;

% Plot the histograms of projected data
figure;
hist(Y1, 20, 'color', 'red');
figure;
hist(Y2, 20, 'color', 'blue');

% Load the dataset
disp('loading digits dataset');
X_train = load('digits_train_features.dat');
Y_train = load('digits_train_labels.dat');
X_test = load('digits_test_features.dat');
Y_test = load('digits_test_labels.dat');
disp('data loaded');

% Form the training data of the two classes
X1_train = X_train(Y_train ==  1, :);
X2_train = X_train(Y_train == -1, :);


% Compute the weights vector
w = fisherLDA(X1_train, X2_train);
figure;
imagesc(reshape(w, 20, 14));

% Project both datasets X1 and X2 on the line
% which is defined by w
[Y1, Y2] = project(w, X1_train, X2_train);

% Estimate w0
w0 = estimateW0(Y1, Y2);

% Evaluate the classifier
disp('evaluating classifier');
y = evaluateFisher(X_test, w, w0);

% Find the accuracy of the classifier
result = find(y ~= Y_test');
ratio = length(result) / length(Y_test);
fprintf('Accuracy: %g\n', ratio);

% Second dataset
% Load the dataset
disp('loading second digits dataset');
X_train = load('digits2_train_features.dat');
Y_train = load('digits2_train_labels.dat');
X_test = load('digits2_test_features.dat');
Y_test = load('digits2_test_labels.dat');
disp('data loaded');

% Form the training data of the two classes
X1_train = X_train((Y_train ==  1), :);
X2_train = X_train((Y_train == -1), :);

% Compute the weights vector
w = fisherLDA(X1_train, X2_train);
figure;
imagesc(reshape(w, 22, 16));

% Project both datasets X1 and X2 on the line
% which is defined by w
[Y1, Y2] = project(w, X1_train, X2_train);

% Estimate w0
w0 = estimateW0(Y1, Y2);

% Evaluate the classifier
y = evaluateFisher(X_test, w, w0);

% Find the accuracy of the classifier
disp('evaluating classifier');
result = find(y ~= Y_test');
ratio = length(result) / length(Y_test);
fprintf('Accuracy: %g\n', ratio);

