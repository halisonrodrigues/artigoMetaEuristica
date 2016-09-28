function [] = test_standard()
%% Loading dataset
clear;
tic
%load orl;
%load yale;
%load umist;
%disp(dataset_name);
%load base\\yeast;
%load base\\classYeast;
load base\\ionosphere2;
load base\\classIonosphere;
disp(dataset_name);

%% Experiment Parameters
test_sample_proportion = .5;
number_of_repetitions = 10;

 %% Classifiers
 classifiers{1} = @ClassificationKNN.fit;
 classifiers{2} = @(training, class)NaiveBayes.fit(training, class, 'Distribution', 'kernel');
 classifiers{3} = @ClassificationDiscriminant.fit;
 classifiers{4} = @ClassificationTree.fit;
 
 classifier1 = classifiers{1};
 classifier2 = classifiers{2};

 %% Teste HOLDOUT
 holdout = cvpartition(Y, 'HoldOut', test_sample_proportion);

results = zeros(number_of_repetitions, 1);
results2 = zeros(number_of_repetitions, 1);
resultsM = zeros(number_of_repetitions, 1);
results2M = zeros(number_of_repetitions, 1);

for iRepetition = 1:number_of_repetitions
    % Holdout partitioning
%     if exist('holdouts_100', 'var')
%        holdout = holdouts_100{iRepetition};
%     else
         holdout = repartition(holdout);
%     end
    disp([ num2str(iRepetition),'/10']);
%    X_NEW = [];
%    for i=1:size(X,1)
%        temp   = reshape(X(i,:),112,92);
%        temp2  = imresize(temp, 1/8);
%        X_NEW  = [X_NEW;reshape(temp2,1,[])];  
%    end
    %disp(size(temp2));
    
    %training and test sets
    training = dataset_name(holdout.training(1),:);
    trainingLabels = Y(holdout.training(1),:);
    
    [si,k,medoids,clusters,tabela_correlacoes] = getClusters(training);
    
    [resultados, kM, novaMedoids,clustersM] = metaHeuristica(si,k,medoids,clusters,tabela_correlacoes,size(dataset_name,2));
      
    n_features(iRepetition) = k; % insere o numero k
    
    treino=training(:,medoids); % insere as medoids
    
    treinoM=training(:,novaMedoids); % treino com as medoids da metaHeuristica
    
    test = dataset_name(holdout.test(1),: );
    testLabels = Y(holdout.test(1),:);
    
    %%    algortimos de seleção aqui
    %%
    % experimento aqui
    classifier_temp = classifier1(treino,trainingLabels);
    classifier_temp2 = classifier2(treino,trainingLabels);
    
    resp_temp = classifier_temp.predict(test(:,medoids));
    resp_temp2 = classifier_temp2.predict(test(:,medoids));
    
    results(iRepetition) = sum(testLabels == resp_temp)/length(resp_temp);    
    results2(iRepetition) = sum(testLabels == resp_temp2)/length(resp_temp2);    

    % experimento metaHeuristica aqui
    classifier_tempM = classifier1(treinoM,trainingLabels);
    classifier_temp2M = classifier2(treinoM,trainingLabels);
    
    resp_tempM = classifier_tempM.predict(test(:,novaMedoids));
    resp_temp2M = classifier_temp2M.predict(test(:,novaMedoids));
    
    resultsM(iRepetition) = sum(testLabels == resp_tempM)/length(resp_tempM);    
    results2M(iRepetition) = sum(testLabels == resp_temp2M)/length(resp_temp2M);
end

%% Results
disp('KNN')
std_accuracy = std(results)
mean_accuracy = mean(results)

std_accuracy = std(resultsM)
mean_accuracy = mean(resultsM)

disp('NB')
std_accuracy = std(results2)
mean_accuracy = mean(results2)

std_accuracy = std(results2M)
mean_accuracy = mean(results2M)

disp(n_features);
disp(['Mean_feature  >>> ' num2str(mean(n_features))]);

toc
end