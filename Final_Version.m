clear
clc

% sample categorations we use to test
categories = {'airplanes', 'cup' , 'camera'};
image_path = '.../101_ObjectCategories/';

train_set = cell(105,1); % plan to use 35 training sample each categories
train_label = cell(105,1);

test_set = cell(45,1); % plan to use 15 training sample each categories
test_label = cell(45,1);


All_data = [imageSet(fullfile('101_ObjectCategories', categories{1})), ...
    imageSet(fullfile('101_ObjectCategories', categories{2})), ...
    imageSet(fullfile('101_ObjectCategories', categories{3}))];

divide = partition(All_data, 50, 'randomized'); % the smallest dataset is 50
[training, test] = partition(divide, 0.7, 'randomized'); % divide the training and test data

% training set(not sure why can't use 3 for loops, it will cause mistakes)
for i = 1:35
    for k = 1:35
        train_set{i,1} = cellstr(training(1,1).ImageLocation{1,k});
        train_label{i,1} = training(1,1).Description;
    end
end
for i = 36:70
    for k = 1:35
            train_set{i,1} = cellstr(training(1,2).ImageLocation{1,k});
            train_label{i,1} = training(1,2).Description;
    end
end
for i = 71:105
    for k = 1:35
            train_set{i,1} = cellstr(training(1,3).ImageLocation{1,k});
            train_label{i,1} = training(1,3).Description;
    end
end

% test set
for i = 1:15
    for k = 1:15
        test_set{i,1} = cellstr(test(1,1).ImageLocation{1,k});
        test_label{i,1} = test(1,1).Description;
    end
end
for i = 16:30
    for k = 1:15
        test_set{i,1} = cellstr(test(1,2).ImageLocation{1,k});
        test_label{i,1} = test(1,2).Description;
    end
end
for i = 31:45
    for k = 1:15
        test_set{i,1} = cellstr(test(1,3).ImageLocation{1,k});
        test_label{i,1} = test(1,3).Description;
    end
end

% flatten cell array
train_set = vertcat(train_set{:});
test_set = vertcat(test_set{:});

% try to get all features for train set
for i = 1: 105
    I = imread(train_set{i,1});
    [~,~,z] = size(I);
    % if not a RGB type, convert to RGB
    if z == 1
        I = repmat(I,[1, 1, 3]);
    end
    img = rgb2gray(I);
    points = detectSIFTFeatures(img);
    [des_matrix, ~] = extractFeatures(img, points);
    [idx, C] = Kmeans(des_matrix, 50,100);
    trainFeatures(i,:)= histcounts(idx, 1:50,'Normalization','pdf')';
end

for i = 1: 45
    I = imread(test_set{i,1});
    [~,~,z] = size(I);
    % if not a RGB type, convert to RGB
    if z == 1
        I = repmat(I,[1, 1, 3]);
    end
    img = rgb2gray(I);
    points = detectSIFTFeatures(img);
    [des_matrix, ~] = extractFeatures(img, points);
    [idx, C] = Kmeans(des_matrix, 50,100);
    testFeatures(i, :)= histcounts(idx, 1:50,'Normalization','pdf')';
end

categories=unique(train_label); 
lambda=0.0001;
scores=[];

for i=1:length(categories)
    w=[];
    b=[];
    match=strcmp(categories(i) , train_label);
    match=double(match);
    for j=1:size(train_label,1)
        if(match(j)==0)
            match(j)=-1;
        end
    end

    % Train SVM
    [w_primal]=train_svm_primal(trainFeatures, match, lambda);
    scores=[scores; (w_primal'*testFeatures')];
end

[~, idx]=max(scores);
prediction=categories(idx');

confusion_mat = zeros(length(categories), length(categories));
for i=1:length(prediction)
    row=find(strcmp(test_label{i}, categories));
    column=find(strcmp(prediction{i}, categories));
    confusion_mat(row, column)=confusion_mat(row, column)+1;
end

num_test_per_cat=length(test_label)/length(categories);
confusion_mat=confusion_mat./num_test_per_cat;   
disp(confusion_mat);
fprintf('Accuracy is %.2f\n', mean(diag(confusion_mat)))




