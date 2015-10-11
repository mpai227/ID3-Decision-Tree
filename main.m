
clc; clear all; close all hidden; 

oracle_number = 7;  %which oracle to use
n_train = 1000;       %number of training examples
n_test = 100;       %number of test examples

[train_attrib,train_class] = trog_DataManager.getTrainingData(oracle_number,n_train);

my_dt = DecisionTree();
my_dt.train(train_attrib,train_class);

% Request testing examples
test_attrib = trog_DataManager.getTestData(oracle_number, n_test);

% Classify the emotions of the test examples from trog-win.exe
test_class = my_dt.classify(test_attrib);

[error_rate,correct_class] = trog_DataManager.checkAccuracy(oracle_number,test_class,n_test);

fprintf('-------------------------- Result ----------------------------\n');
fprintf('Oracle \t #Training Sample \t #Test Sample \t Error Rate (%%)\n');
fprintf('%6d \t %16d \t %12d \t %4.2f\n', oracle_number, n_train, n_test, error_rate);

% visualize the tree
my_dt.plot();
