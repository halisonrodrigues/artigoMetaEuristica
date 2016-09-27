function [SSV,k_next,medoids_save_OUT,clusters_save_OUT,table_correlations] = getClusters(data)
   tic
    kmin = 2;
    n_partitions = 20;
    SSV = -Inf;
        
    %%      CALCULO DE CORRELAÇÕES
    
    table_correlations = ones(size(data,2));
    for i = 1:(size(data,2))
        feature1 = data(:,i);
        for j = i:size(data,2)
            feature2 = data(:,j);
            coefficient = cov(feature1,feature2)/sqrt(var(feature1)*var(feature2));
            correlation = (var(feature1) + var(feature2)) - (sqrt( (var(feature1) + var(feature2))^2 - 4 * var(feature1) * var(feature2) * ( 1 - (str2num(num2str(coefficient(1,2))))^2) ) );
            table_correlations(i,j) = correlation;
            table_correlations(j,i) = correlation; 
        end
    end

    %%
    for i = kmin:(size(data,2)-1)
        %disp(['k >>> ', int2str(i)]);
        n_features_cluster = floor(size(data,2)/ double(i));
        n_features_cluster_plus = mod(size(data,2),i);
        clusters_save = {};
        medoids_save  = [];
        BOV = -Inf;
        for n = 1:n_partitions
            
            [positions,~] = shuffle(data);
            clusters = {};
            ini = 1;
            fim = n_features_cluster;
            contador_plus = n_features_cluster_plus;            
            %Criação aleatória dos clusters
            for j = 1:i
                if(contador_plus~=0)
                    clusters{j} = positions(1,ini:(fim+1));
                    fim = fim+1;
                    ini = fim + 1;
                    fim = fim + n_features_cluster;
                    contador_plus = contador_plus - 1;
                else
                    clusters{j} = positions(1,ini:fim);
                    ini = fim + 1;
                    fim = fim + n_features_cluster;
                end                
            end
            
            %k-medoids
            clusters_KM = clusters;
            while (true)
                %Obtendo as medoids
                medoids_k = [];
                for j = 1:i
                    correlations_k = sum(table_correlations(:,clusters_KM{j}));
                    medoids_k = [medoids_k,min(clusters_KM{j}(min(correlations_k) == correlations_k))];
                end  
                for r = 1:numel(clusters)
                    
                    medoid = medoids_k(r);
                    for o = 1:size(clusters{r},2)    
                        feature = clusters{r}(o);
                        if(medoid==feature)
                           continue 
                        end
                        corralation_k = [];
                        for t = 1:size(medoids_k,2)
                            corralation_k = [corralation_k,table_correlations(feature,medoids_k(t))];
                        end
                        %Verificando se a feature não está no cluster com a medoid
                        if (sum(feature == clusters{min(corralation_k)==corralation_k}) == 0 )
                            
                            %clusters_KM{min(corralation_k)==corralation_k} = [clusters_KM{min(corralation_k)==corralation_k},feature];
                            %clusters_KM{r}(clusters_KM{r}==feature) = [];
                            
                            %INDICE = find(min(corralation_k)==corralation_k);
                            %clusters_KM{INDICE(1)} = [clusters_KM{min(corralation_k)==corralation_k},feature];
                            if sum(clusters_KM{min(corralation_k)==corralation_k}==feature)~=1
                                clusters_KM{min(corralation_k)==corralation_k} = [clusters_KM{min(corralation_k)==corralation_k},feature];
                                clusters_KM{r}(clusters_KM{r}==feature) = [];
                            end
                            
                            
                        end
                    end
                end
                
                medoids_k_aux = [];
                for j = 1:i
                    correlations_k = sum(table_correlations(:,clusters_KM{j}));
                    medoids_k_aux = [medoids_k_aux,min(clusters_KM{j}(min(correlations_k) == correlations_k))];
                end
                %Verifica se se as medoids são consecutivas
                if(sum(medoids_k == medoids_k_aux)==size(medoids_k_aux,2))
                    break;
                end
                
            end
            
            clusters = clusters_KM;
            
            %Simplifeid Silhouette
            si_partition = [];
            for m = 1:numel(clusters)
                
                if size(clusters{m},2) == 1
                    si_partition = [si_partition, 0];
                    continue;
                end
                
                medoid = medoids_k(m);
                for o = 1:size(clusters{m},2)
                    
                    feature = clusters{m}(o);
                    ai = table_correlations(feature,medoid);
                    
                    %Buscar a dissimilaridade para o cluster mais próximo - bi
                    bi = Inf;
                    for t = 1:size(medoids_k,2)
                        if (medoids_k(t) == medoid)
                            continue;
                        end
                        bi = min(bi,table_correlations(feature,medoids_k(t)));
                    end
                    si = (bi - ai) / max(ai,bi);
                    si_partition = [si_partition, si];
        
                end
                
            end
            
            mean_si = mean(si_partition);
            
            if (mean_si>BOV)
                clusters_save = clusters;
                medoids_save  = medoids_k;
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
    toc
end

function [pos,data_set] = shuffle(data_set)
    data_set = data_set(1:end,1:(size(data_set,2)));
    pos   = randperm(size(data_set,2));
    data_set = data_set(:,pos);
end