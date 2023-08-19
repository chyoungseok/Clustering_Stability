import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def explore_new_label(dist_btween_centers, nk):
    """

    """
    
    all_false = [False] * nk
    new_label = np.ones(nk) * -1 # B clustering의 기존 label을 어떤 label로 바꿔야 할지에 대한 정보
    # new_label[0] = 1 인 경우, 0번 cluster는 1번 으로 label 변경
    
    for i in range(nk):
        min_idx = dist_btween_centers.argmin() # 전체 value 중, minimum value의 location
        min_idx_i = min_idx // nk # min value가 위치한 row (i th cluster label in bootstrap data)
        min_idx_j = min_idx - min_idx_i * nk # min value가 위치한 column (j th cluster label in original data)
        
        if min_idx_j in new_label:
            # min_idx_j가 이미 등장한 경우, inf로 설정해줌으로써 min에 의해 탐지되지 않도록 함
            dist_btween_centers[min_idx_i, min_idx_j] = np.inf
            min_idx = dist_btween_centers.argmin()
            min_idx_i = min_idx // nk
            min_idx_j = min_idx - min_idx_i * nk
        
        new_label[min_idx_i] = min_idx_j
        
        # i-th cluster of boostrap data는 이미 배정됐으니, min 값 탐색 후보에서 제외
        # min 값 탐색 후보에서 제외하기 위하여 inf로 설정
        all_false[min_idx_i] = True
        dist_btween_centers[all_false] = np.inf 
    
    return new_label.astype(int)

def map_B_center_to_O_center(org_data, O_centers, O_lables, B_centers, B2O_mapping_method, nk):
    """
        Bootstrap data로부터 구한 center를 Original data의 center에 matching 시키기
        
        sim_method에 따라서 mapping method가 다름
    """
    if B2O_mapping_method == 'euclidean':
        distances = cdist(B_centers, O_centers, metric='euclidean') # distance matrix between two centers
        # for row i and column j,
        # [i, j] --> distance between i_th B_center and j_th O_center
        dist_btween_centers = distances.copy() # distances 복제
        
        
    elif B2O_mapping_method == 'jaccard':
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
        dist_btween_centers = distances.copy()
                
    new_label = explore_new_label(dist_btween_centers = dist_btween_centers, nk = nk)
    new_center = B_centers[new_label]
            
    return  new_center

def cal_jaccard(org_set, o2b_set):
    return float(len(org_set.intersection(o2b_set)) / len(org_set.union(o2b_set)))    

class clustering_algs():
    def __init__(self, data, clst_alg, K, random_state=None) -> None:
        self.clst_alg = clst_alg
        
        if clst_alg == 'kmeans':
            _km = KMeans(n_clusters=K, random_state=random_state, init="k-means++")
            _km.fit(data)
            
            self.center = _km.cluster_centers_
            self.B2O_center = _km.cluster_centers_ * 0 # _bootClustering.B2O_center = mapped_center  을 통해 업데이트 되어야 함
            # stability_scheme1.py의 line88 참고
            self.labels = _km.labels_        
            self.data = data 
