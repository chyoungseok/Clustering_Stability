import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from modules.utils import clustering_algs, map_B_center_to_O_center, cal_jaccard
from scipy.spatial.distance import cdist


class stability():
    """ Input parameters
        - org_data : an original dataset
        - K : number of clusters (default=2)
        - clsg_alg : an algorithm to perform clustering (default="kmeans")
        - B : a total number of the bootstrapping (B=10)
        - sim_method : how to measure a similarity between samples (default="euclidean")
    """
    
    @staticmethod
    def getStabilities(stability_matrix, _orgClustering, K):
        '''
            input: stability matrix; shape=(B,n); B=number of bootstrapping; n=number of data
            
            output: list of stabilities
                [observation_wise, cluster_wise, overall]
        '''
        observation_wise = np.mean(stability_matrix, axis=0)
        cluster_wise = []
        for ki in range(K):
            ki_clust_stab = observation_wise[_orgClustering.labels == ki]
            cluster_wise.append(ki_clust_stab.mean())
        overall = np.mean(observation_wise)
        return [observation_wise, cluster_wise, overall]
    
    @staticmethod
    def plot_K_optimization(data, max_K=9, B=5):
        Smins = []
        for k in range(2, max_K+1):
            _stability = stability(org_data=data, K=k, B=B)
            Smins.append(_stability.Smin)
            
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        plt.plot(range(2,max_K+1), Smins, marker='o', label="S$_{min}$");
        plt.legend()
        plt.ylabel("S$_{min}$", fontdict={'fontsize':15})
        plt.xlabel("K (number of clusters)", fontdict={'fontsize':15})
        plt.title("Optimize the K (Bootstrapping=%d times for each K)" % B)
        ax.set_xticks(range(2,max_K+1))
            
    
    def __init__(self, org_data, K=2, B=10, B2O_mapping_method='jaccard', clst_alg='kmeans') -> None:
        self.org_data = org_data
        self.K = K
        self.B = B
        # self.sim_method = sim_method
        self.clst_alg = clst_alg
        self.B2O_mapping_method = B2O_mapping_method

        # Clustering for the original data
        _orgClustering = clustering_algs(data = org_data,
                                         clst_alg = clst_alg,
                                         K = K)
                                        #  sim_method = sim_method)
        
        # Clustering for the each of the bootstrapped data
        list_bootClustering = []
        stability_matrix_naive = np.empty((B, org_data.shape[0]))
        stability_matrix_jaccard = np.empty((B, org_data.shape[0]))
        stability_matrix_cluster_wise_jaccard = np.empty((B, K))
        for b in tqdm(range(B), desc="Bootstrapping..."):
            resample_idx = np.random.choice(org_data.shape[0], size=org_data.shape[0], replace=True)
            boot_data = org_data[resample_idx, ]
            
            # clustering for the bootstrapped data
            _bootClustering = clustering_algs(data = boot_data,
                                              clst_alg = clst_alg,
                                              K = K,
                                              random_state = b)
                                            #   sim_method = sim_method,
                                              
            
            # mapping B_centers into O_centers
            mapped_center = map_B_center_to_O_center(org_data=org_data,
                                                     O_centers = _orgClustering.center,
                                                     O_lables = _orgClustering.labels,
                                                     B_centers = _bootClustering.center,
                                                     B2O_mapping_method = B2O_mapping_method,
                                                     nk = K)
            _bootClustering.B2O_center = mapped_center # B2O_center update          
            list_bootClustering.append(_bootClustering)
            
            o2b_labels = cdist(_bootClustering.center, org_data, metric='euclidean').argmin(axis=0) # original data를 bootClustering의 center에 mapping
            mapped_o2b_labels = cdist(mapped_center, org_data, metric='euclidean').argmin(axis=0) # original data를 mapped bootClustering의 center에 mapping
            idx = np.arange(org_data.shape[0])
            
            # get stability matrix for naive and jaccard based methods
            for i in range(org_data.shape[0]):
                temp_org_label = _orgClustering.labels[i] # x_i가 origianl clustering에서 가지는 label
                temp_o2b_label = o2b_labels[i] # x_i가 boot clustering에서 가지는 label
                temp_mapped_o2b_label = mapped_o2b_labels[i] # x_i가 mapped boot clustering에서 가지는 label
                
                org_set = set(idx[_orgClustering.labels == temp_org_label]) # x_i와 같은 orgCluster에 있는 data members
                o2b_set = set(idx[o2b_labels == temp_o2b_label]) # x_i와 같은 bootCluster에 있는 data members
                mapped_o2b_set = set(idx[mapped_o2b_labels == temp_mapped_o2b_label]) # x_i와 같은 o2bCluster에 있는 data members
                
                # Naive stability
                if len(org_set.intersection(mapped_o2b_set)) == len(org_set):
                    stability_matrix_naive[b, i] = 1
                else:
                    stability_matrix_naive[b, i] = 0
                    
                # Jaccard based stability
                stability_matrix_jaccard[b, i] = cal_jaccard(org_set, o2b_set)
                
            
            # Smin (scheme1)
            # Cluster-wise jaccard based stability 
            for k in range(K):
                cluster_wise_similarity = 0
                org_set = set(idx[_orgClustering.labels == k]) # k 번째 org cluster에 포함된 모든 sample들의 index
                for idx_for_tempK in org_set: # org_list의 모든 sample에 대한 반복문
                    temp_label = o2b_labels[idx_for_tempK] # temp sample의 o2b_label
                    o2b_set = set(idx[o2b_labels == temp_label]) # temp sample이 포함되어 있는 o2b cluster의 모든 sample의 index
                    
                    cluster_wise_similarity += cal_jaccard(org_set, o2b_set)
                
                stability_matrix_cluster_wise_jaccard[b, k] = cluster_wise_similarity/len(org_set)

        Smin_scheme1 = np.mean(np.min(stability_matrix_cluster_wise_jaccard, axis=1)) # calculate Smin

        # Smin (scheme2)
        # reference clustering을 original clustering 뿐만 아니라, 각각의 bootstrapping clustering에도 적용
        Smin_scheme2_for_each_ref = np.empty((B+1, 1)) # reference clustering을 달리하면서 계산한 각 Smin을 기록 --> 마지막에 이들을 평균내서 Smin_scheme2를 구함
        total_clustering = [_orgClustering] + list_bootClustering

        for r in range(B+1):
            # total_clustering의 모든 요소가 reference_clustering의 후보임 --> 각 ref에 대하여 반복
            ref_clustering = total_clustering[r] # reference clustering 지정
            stability_matrix_cluster_wise_jaccard_scheme2 = np.empty((B, K)) # 현재 ref_clustering에 대한 cluster wise stability를 저장할 matrix

            b_iter = 0
            for b in range(B+1):
                if b == r:
                    continue
                
                r2b_labels = cdist(total_clustering[b].center, total_clustering[r].data, metric='euclidean').argmin(axis=0) # reference data를 bootClustering의 center에 mapping

                # Cluster-wise jaccard based stability (comparison between ref_clustering vs. total_clustering[b])
                for k in range(K):
                    cluster_wise_similarity = 0
                    ref_set = set(idx[ref_clustering.labels == k]) # ref_clustering의 k번째 cluster에 포함된 모든 sample들의 index
                        
                    for idx_for_tempK in ref_set: # ref_idx의 모든 sample에 대한 반복문
                        temp_label = r2b_labels[idx_for_tempK] # temp sample의 r2b_label
                        r2b_set = set(idx[r2b_labels == temp_label]) # temp sample이 포함되어 있는 r2b cluster의 모든 sample의 index
                        cluster_wise_similarity += cal_jaccard(ref_set, r2b_set)
                    stability_matrix_cluster_wise_jaccard_scheme2[b_iter, k] = cluster_wise_similarity/len(ref_set)
                b_iter += 1
            Smin_scheme2_for_each_ref[r] = np.mean(np.min(stability_matrix_cluster_wise_jaccard_scheme2, axis=1)) # calculate Smin for each reference clustering
        Smin_scheme2 = np.mean(Smin_scheme2_for_each_ref) # Smin_scheme2 calculated by averaging all the Smin of reference clustering


            
        self._orgClustering = _orgClustering
        self.list_bootClustering = list_bootClustering
        self.stability_matrix_naive = stability_matrix_naive
        self.stability_matrix_jaccard = stability_matrix_jaccard
        self.stability_matrix_cluster_wise_jaccard = stability_matrix_cluster_wise_jaccard
        self.naive_stabs = self.getStabilities(stability_matrix_naive, _orgClustering, K=K)
        self.jaccard_stabs = self.getStabilities(stability_matrix_jaccard, _orgClustering, K=K)
        self.Smin_scheme1 = Smin_scheme1
        self.Smin_scheme2 = Smin_scheme2