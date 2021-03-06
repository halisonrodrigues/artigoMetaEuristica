function [resultados,k,novaMedoids,clusters] = metaHeuristica(si,k,medoids,clusters,tabela_correlacoes,tamanho)
%% c�lculo do algoritmo original
%load ('base\\yeast.mat');
%[si,k,medoids,clusters,tabela_correlacoes] = getClusters(dataset_name);
antigaSi = si;
nv = 1;
v1 = true;
contadorShaking = 1;
resultados = {1,10};
%% Metaheur�stica
clustersOriginal = clusters;
% seleciona os x candidatos de acordo com a tabela de correla��o
for epocas=1:10
candidatos = getCandidatos(k,medoids,clustersOriginal,tabela_correlacoes);
% Muda as medoids dos clusters (entre os x candidatos mais correlacionados)
% Calcula o Si com as novas medoids (dentro da fun��o vizinhanca1)
novaMedoids = medoids;
    while (nv < 4)
        if (nv == 1)
            [novaSi,novaMedoids] = vizinhanca1(candidatos, novaMedoids, clusters,tabela_correlacoes,si);
            while (v1)
                if (antigaSi >= novaSi)
                    v1 = false;
                    nv = 2;
                else
                    antigaSi = novaSi;
                    candidatos = getCandidatos(k,novaMedoids,clusters,tabela_correlacoes);
                    novaSi = vizinhanca1(candidatos,novaMedoids,clusters,tabela_correlacoes,novaSi);
                end
            end
            % Se n�o melhorar o Si troca uma caracter�stica de cluster (entre os x candidatos mais correlacionados)
            % Calcula o Si com os novos clusters
        elseif (nv == 2)
            candidatos = getCandidatos(k,novaMedoids,clusters,tabela_correlacoes);
            novaSi2 = vizinhanca2(clusters,candidatos,novaMedoids,tabela_correlacoes);
            if (antigaSi < novaSi2)
                antigaSi = novaSi2;
            end
            nv = 3;
            % Ap�s as duas vizinhan�as faz-se o shaking para evitar os �timos
            % locais. Este � feito retirando uma caracter�stica aleat�ria do
            % cluster e a colocando em outro cluster
        elseif (nv == 3)
            if (contadorShaking < 10)
                [clusters,novaMedoids] = shaking(clusters,tamanho,novaMedoids,candidatos);
                candidatos = getCandidatos(k,novaMedoids,clusters,tabela_correlacoes);
                nv = 1;
                v1 = true;
                contadorShaking = contadorShaking + 1;
            else
                nv = 4;
            end
        end
    end
    saida = [si,antigaSi];
    resultados{1,epocas} = saida;
    resultados{2,epocas} = novaMedoids;
end

end
%% sele��o de candidatos
function [candidatos] = getCandidatos(k,medoids,clusters,tabela_correlacoes)
candidatos = [];
for i=1:k
   if (size(clusters{1,i},2) <= 6)
       for j=1:size(clusters{1,i},2)
           if (clusters{1,i}(1,j) ~= medoids(1,i))
               candidatos = [candidatos,clusters{1,i}(1,j)];
           end
       end
   else
       clusterCorrelate = [];
       for j=1:size(clusters{1,i},2)
           if (clusters{1,i}(1,j) ~= medoids(1,i))
               soma = sum(tabela_correlacoes(clusters{1,i}(1,j),:));
               clusterCorrelate = [clusterCorrelate,soma];
           else
               clusterCorrelate = [clusterCorrelate,0];
           end
       end
       for k=1:5
           id = find(clusterCorrelate == max(clusterCorrelate));
           candidatos = [candidatos,clusters{1,i}(1,id)];
           clusterCorrelate(1,id) = 0;
       end
   end
end
end

%% Simplifeid Silhouette
function [SSV] = silhueta(clusters, medoids,tabela_correlacoes)
SSV = -Inf;
BOV = -Inf;
si_partition = [];
for m = 1:numel(clusters)

    if size(clusters{m},2) == 1
        si_partition = [si_partition, 0];
        continue;
    end

    medoid = medoids(m);
    for o = 1:size(clusters{m},2)

        feature = clusters{m}(o);
        ai = tabela_correlacoes(feature,medoid);

        %Buscar a dissimilaridade para o cluster mais pr�ximo - bi
        bi = Inf;
        for t = 1:size(medoids,2)
            if (medoids(t) == medoid)
                continue;
            end
            bi = min(bi,tabela_correlacoes(feature,medoids(t)));
        end
        si = (bi - ai) / max(ai,bi);
        si_partition = [si_partition, si];

    end



    mean_si = mean(si_partition);

    if (mean_si>BOV)
        clusters_save = clusters;
        medoids_save  = medoids;
        BOV = mean_si;
    end

end

if (BOV > SSV)
    SSV = BOV;
    k_next = i;
    medoids_save_OUT = medoids_save;
    clusters_save_OUT = clusters_save;
end  
end

%% Mudan�a de medoid
function [siAtual,medoids] = vizinhanca1(candidatos, medoids, clusters, tabela_correlacoes,siAtual)
novosCandidatos = candidatos;
x = 1;
while x > 0
    x = sum(novosCandidatos);
    for i=1:size(medoids,2)
       if (size(clusters{1,i},2) > 1)
          for j=1:size(clusters{1,i},2)
             if (sum(clusters{1,i}(1,j) == novosCandidatos) == 1)
                medoids(1,i) = clusters{1,i}(1,j);
                novosCandidatos(1,find(novosCandidatos == clusters{1,i}(1,j))) = 0;
                break;
             end
          end
       end
       continue;
    end
    novaSi =  silhueta(clusters, medoids,tabela_correlacoes);
    if (novaSi > siAtual)
        siAtual = novaSi;
    end
end
end

%% mudando a caracteristica de cluster
function [si2] = vizinhanca2(clusters, candidatos,medoids,tabela_correlacoes)
novosClusters = clusters;
si2 = -1;
for c=1:size(novosClusters,2)
    clusterAtual = novosClusters{1,c};
    for a=1:size(clusterAtual,2)
        if (sum(clusterAtual(1,a) == candidatos) == 1)
            shiftCandidado = clusterAtual(a);
            novosClusters{1,c}(a) = [];
            for b=1:size(novosClusters,2)
                if b ~= c
                    novosClusters{1,b} = [novosClusters{1,b}, shiftCandidado];
                    novaSi = silhueta(novosClusters, medoids,tabela_correlacoes);
                    si2 = max(si2,novaSi);
                    novosClusters{1,b}(size(novosClusters{1,b},2)) = [];
                end
            end
            novosClusters{1,c} = [novosClusters{1,c},shiftCandidado];
        end
    end 
end
end

%% shaking
function [novosClusters, medoids] = shaking(clusters,tamanhoBase,medoids,candidatos)
novosClusters = clusters;
numeroFeature = randi(tamanhoBase);
numeroCluster = randi(size(clusters,2));
numeroClusterRetirado = 0;
for i=1:size(clusters,2)
    if (sum(clusters{1,i} == numeroFeature) == 1 && size(clusters{1,i},2 > 1))
        if (sum(medoids == numeroFeature) == 0)
            indiceFeature = find(clusters{1,i} == numeroFeature);
            novosClusters{1,i}(indiceFeature) = [];
            numeroClusterRetirado = i;
            break;
        else
            for c=1:size(candidatos,2)
                if (sum(clusters{1,i} == candidatos(1,c)) == 1)
                    novaMed = candidatos(1,c);
                    indiceMedoid = find(medoids == numeroFeature);
                    medoids(1,indiceMedoid) = novaMed;
                    indiceFeature = find(clusters{1,i} == numeroFeature);
                    novosClusters{1,i}(indiceFeature) = [];
                    numeroClusterRetirado = i;
                    break;
                end
            end
            break;
        end
    end
end
condicao = 1;
while (condicao)
    if (numeroClusterRetirado == numeroCluster)
        numeroCluster = randi(size(clusters,2));
    else
        novosClusters{1,numeroCluster} = [novosClusters{1,numeroCluster}, numeroFeature];
        condicao = 0;
    end
end
end