import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def explore_new_label(distances_rest, nk):
    all_false = [False] * nk
    new_label = np.ones(nk) * -1 # B clustering의 기존 label을 어떤 label로 바꿔야 할지에 대한 정보
    # new_label[0] = 1 인 경우, 0번 cluster는 1번 으로 label 변경
    
    for i in range(nk):
        min_idx = distances_rest.argmin() # 전체 value 중, minimum value의 location
        min_idx_i = min_idx // nk # min value가 위치한 row (i th cluster label in bootstrap data)
        min_idx_j = min_idx - min_idx_i * nk # min value가 위치한 column (j th cluster label in original data)
        
        if min_idx_j in new_label:
            # min_idx_j가 이미 등장한 경우, inf로 설정해줌으로써 min에 의해 탐지되지 않도록 함
            distances_rest[min_idx_i, min_idx_j] = np.inf
            min_idx = distances_rest.argmin()
            min_idx_i = min_idx // nk
            min_idx_j = min_idx - min_idx_i * nk
        
        new_label[min_idx_i] = min_idx_j
        
        # i-th cluster of boostrap data는 이미 배정됐으니, min 값 탐색 후보에서 제외
        # min 값 탐색 후보에서 제외하기 위하여 inf로 설정
        all_false[min_idx_i] = True
        distances_rest[all_false] = np.inf 
    
    return new_label.astype(int)

def map_B_center_to_O_center(org_data, O_centers, O_lables, B_centers, sim_method, nk):
    """
        Bootstrap data로부터 구한 center를 Original data의 center에 matching 시키기
        
        sim_method에 따라서 mapping method가 다름
    """
    if sim_method == 'euclidean':
        distances = cdist(B_centers, O_centers, metric='euclidean') # distance matrix between two centers
        # for row i and column j,
        # [i, j] --> distance between i_th B_center and j_th O_center
        distances_rest = distances.copy() # distances 복제
        
        
    elif sim_method == 'jaccard':
         # original data와 B_centroids 중, 가장 거리가 짧은 centroid가 속한 cluster로 labeling
        o2b_labels = cdist(B_centers, org_data, metric='euclidean').argmin(axis=0)
        
        distances = np.ones((nk,nk)) # 두 클러스터 간 jaccard coefficients를 기록하는 matrix
        idx = np.arange(org_data.shape[0]) # data크기에 해당하는 index array
        for i in range(nk):
            for j in range(nk):
                O_set = set(idx[O_lables == i]) # cluster label이 i인 index from original clustering label
                o2b_set = set(idx[o2b_labels == j]) # cluster label이 j인 index from ob2 clustering label
                
                intersection = len(O_set.intersection(o2b_set)) # 교집합
                union = len(O_set.union(o2b_set)) # 합집합
                
                jaccard = float(intersection / union) # jaccard coefficients
                
                distances[i, j] = -jaccard # matrix update
                # minimum 값을 찾아가는 explore_new_label() 알고리즘에 맞추기 위하여 (-)를 취해줌
                # jaccard는 값이 클수록 similarity가 크기 때문
        distances_rest = distances.copy()
                
    new_label = explore_new_label(distances_rest = distances_rest, nk = nk)
    new_center = B_centers[new_label]
            
    return  new_center

def cal_jaccard(org_set, o2b_set):
    return float(len(org_set.intersection(o2b_set)) / len(org_set.union(o2b_set)))    

class clustering_algs():
    def __init__(self, data, clst_alg, K, sim_method, random_state=None) -> None:
        self.clst_alg = clst_alg
        
        if clst_alg == 'kmeans':
            _km = KMeans(n_clusters=K, random_state=random_state)
            _km.fit(data)
            
            self.center = _km.cluster_centers_
            self.maaped_center = _km.cluster_centers_ * 0
            self.labels = _km.labels_         

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
            
    
    def __init__(self, org_data, K=2, B=10, sim_method="euclidean", clst_alg='kmeans') -> None:
        self.org_data = org_data
        self.K = K
        self.B = B
        self.sim_method = sim_method
        self.clst_alg = clst_alg

        # Clustering for the original data
        _orgClustering = clustering_algs(data = org_data,
                                         clst_alg = clst_alg,
                                         K = K,
                                         sim_method = sim_method)
        
        # Clustering for the each of the bootstrapped data
        list_bootClustering = []
        stability_matrix_naive = np.empty((B, org_data.shape[0]))
        stability_matrix_jaccard = np.empty((B, org_data.shape[0]))
        stability_matrix_cluster_wise_jaccard = np.empty((B, K))
        for b in range(B):
            resample_idx = np.random.choice(org_data.shape[0], size=org_data.shape[0], replace=True)
            boot_data = org_data[resample_idx, ]
            
            # clustering for the bootstrapped data
            _bootClustering = clustering_algs(data = boot_data,
                                              clst_alg = clst_alg,
                                              K = K,
                                              sim_method = sim_method,
                                              random_state = b)
            
            # mapping B_centers into O_centers
            mapped_center = map_B_center_to_O_center(org_data=org_data,
                                                     O_centers = _orgClustering.center,
                                                     O_lables = _orgClustering.labels,
                                                     B_centers = _bootClustering.center,
                                                     sim_method = sim_method,
                                                     nk = K)
            _bootClustering.o2b_center = mapped_center            
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
                
            
            # Cluster-wise jaccard based stability
            for k in range(K):
                cluster_wise_similarity = 0
                org_set = set(idx[_orgClustering.labels == k]) # k 번째 cluster에 포함된 모든 sample들의 index
                for idx_for_tempK in org_set: # org_list의 모든 sample에 대한 반복문
                    temp_label = o2b_labels[idx_for_tempK] # temp sample의 o2b_label
                    o2b_set = set(idx[o2b_labels == temp_label]) # temp sample이 포함되어 있는 o2b cluster의 모든 sample의 index
                    
                    cluster_wise_similarity += cal_jaccard(org_set, o2b_set)
                
                stability_matrix_cluster_wise_jaccard[b, k] = cluster_wise_similarity/len(org_set)

            Smin = np.mean(np.min(stability_matrix_cluster_wise_jaccard, axis=1)) # calculate Smin
            
        self._orgClustering = _orgClustering
        self.list_bootClustering = list_bootClustering
        self.stability_matrix_naive = stability_matrix_naive
        self.stability_matrix_jaccard = stability_matrix_jaccard
        self.stability_matrix_cluster_wise_jaccard = stability_matrix_cluster_wise_jaccard
        self.naive_stabs = self.getStabilities(stability_matrix_naive, _orgClustering, K=K)
        self.jaccard_stabs = self.getStabilities(stability_matrix_jaccard, _orgClustering, K=K)
        self.Smin = Smin