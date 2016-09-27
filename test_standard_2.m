%% Loading dataset
clear;
tic
load orl;
%load yale;
%load umist;
disp(dataset_name);

%% Experiment Parameters
% test_sample_proportion = .5
number_of_repetitions = 10;

 %% Classifiers
 classifiers{1} = @ClassificationKNN.fit;
 classifiers{2} = @(training, class)NaiveBayes.fit(training, class, 'Distribution', 'kernel');
 classifiers{3} = @ClassificationDiscriminant.fit;
 classifiers{4} = @ClassificationTree.fit;
 
 classifier1 = classifiers{1};
 classifier2 = classifiers{2};

 %% Teste HOLDOUT
% holdout = cvpartition(Y, 'holdout', test_sample_proportion);

results = zeros(number_of_repetitions, 1);
for iRepetition = 1:number_of_repetitions
    % Holdout partitioning
%     if exist('holdouts_100', 'var')
        holdout = holdouts_100{iRepetition};
%     else
%         holdout = repartition(holdout);
%     end
    disp([ num2str(iRepetition),'/10']);
    X_NEW = [];
    for i=1:size(X,1)
        temp   = reshape(X(i,:),112,92);
        temp2  = imresize(temp, 1/8);
        X_NEW  = [X_NEW;reshape(temp2,1,[])];  
    end
    %disp(size(temp2));
    
    %training and test sets
    training = X_NEW(holdout.training(1),:);
    trainingLabels = Y(holdout.training(1),:);
    
    [~,k_next,medoids,features_2,~] = getClusters2(training);
    
    n_features(iRepetition) = k_next;
    
    treino=training(:,[medoids,features_2]);
    
    test = X_NEW(holdout.test(1),: );
    testLabels = Y(holdout.test(1),:);
    
    %%    algortimos de seleção aqui
    %%
    % experimento aqui
    classifier_temp = classifier1(treino,trainingLabels);
    classifier_temp2 = classifier2(treino,trainingLabels);
    
    resp_temp = classifier_temp.predict(test(:,[medoids,features_2]));
    resp_temp2 = classifier_temp2.predict(test(:,[medoids,features_2]));
    
    results(iRepetition) = sum(strcmp(testLabels, resp_temp))/length(resp_temp);    
    results2(iRepetition) = sum(strcmp(testLabels, resp_temp2))/length(resp_temp2);    
                 
end

%% Results
disp('KNN')
std_accuracy = std(results)
mean_accuracy = mean(results)

disp('NB')
std_accuracy = std(results2)
mean_accuracy = mean(results2)

disp(n_features);
disp(['Mean_feature  >>> ' num2str(mean(n_features))]);

toc