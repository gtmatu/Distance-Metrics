import math
import scipy
from scipy import signal
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import numpy as np
from metric_learn import LMNN
from sklearn import preprocessing
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

####################################################
####################################################
# Enter function description here
def function():
    pass

####################################################
###############      Utilities      ################
####################################################

###############################################################
###############################################################
# Split dataset into train and test by separating first train_classes 
# classes into train and remaining classes into the test set
def train_test_split(x_data, y_data, normalize=False, train_classes = 32):
    
    if normalize:
        x_data = preprocessing.normalize(x_data, norm='l2', axis=1)
    
    shapeX = x_data.shape
    shapeY = y_data.shape
    x_train = np.empty(shape=(train_classes*10, shapeX[1]), dtype=int)
    y_train = np.empty(shape=(train_classes*10), dtype=int)
    x_test = np.empty(shape=(int(shapeX[0]-train_classes*10), shapeX[1]), dtype=int)
    y_test = np.empty(shape=(int(shapeX[0]-train_classes*10)), dtype=int)
    
    x_train = x_data[:train_classes*10]
    y_train = y_data[:train_classes*10]
    x_test = x_data[train_classes*10:]
    y_test = y_data[train_classes*10:]
    
    return(x_train, y_train, x_test, y_test)

####################################################
####################################################
# show image from 1-D array
FIGURE_NO = 1
def show_image(pixels, label="", cmap='gray', independent=True, axis=True):
    global FIGURE_NO
    picH = 56
    picW = 46

    if independent:
        plt.figure(FIGURE_NO)
    if not axis:
        plt.axis('off')
    plt.imshow(pixels.reshape(picH, picW), cmap=cmap)
    if label:
        plt.title(label)
    FIGURE_NO += 1
    
####################################################
###################      Q1      ################### 
####################################################

####################################################
####################################################
# Returns Acc@1, Acc@10, mAP for given test set,
# Ensure images are ordered, and labels start from 0
def get_scores(x_data, y_data, metric='euclidean', arg=None):
    acc_1 = np.zeros(shape=(x_data.shape[0]))
    acc_10 = np.zeros(shape=(x_data.shape[0]))
    average_precisions = np.zeros(shape=(x_data.shape[0]))

    for index, query in enumerate(x_data):
        query, test_set = pop(x_data, index=index)
        query_label, test_set_labels = pop(y_data, index=index)
        
        num_neighbors = test_set.shape[0]
        
        if arg is None:
            knn = get_classifier(num_neighbors, metric)
        else:
            knn = get_classifier(num_neighbors, metric, arg=arg)
        knn = knn.fit(test_set, test_set_labels)
        pred = knn.kneighbors(query, return_distance=False)


### Test
#     if arg is None:
#         knn = get_classifier(199, metric)
#     else:
#         knn = get_classifier(199, metric, arg=arg)

#     knn = knn.fit(x_data, y_data)
#     pred = knn.kneighbors(return_distance=False)
    
#     for index, nearest_neighbors in enumerate(pred):
#         query_label = y_data[index]
### Test end

        nearest_neighbors = pred[0]

        # Convert from img number to class number
        for idx, label in enumerate(nearest_neighbors):
            nearest_neighbors[idx] = (label - label%10) / 10
        

        if query_label == nearest_neighbors[0]:
            acc_1[index] = 1
        else:
            acc_1[index] = 0

        if (query_label in nearest_neighbors[:10]):
            acc_10[index] = 1
        else:
            acc_10[index] = 0

        prec_idx = 1 # Start at 1, since P@R(0) <= P@R(0.11), hence upon interpolating the info isn't necessary
        precisions = np.zeros(shape=(10))
        all_neighbours_found = False
        for num_retrieved, latest_retrieved in enumerate(nearest_neighbors, 1):
            
            # Every time a match is found, the recall value goes up (we reach a new recall level), and we calculate 
            # precision at that point (we only keep the maximum precision for said recall level, which is 
            # always the first found)-> $precisions$ is the precision-recall curve
            if (query_label == latest_retrieved):
                nearest_neighbors_trunc = nearest_neighbors[:num_retrieved]
                success_count = list(nearest_neighbors_trunc).count(query_label)
                
                precisions[prec_idx] = success_count / num_retrieved
                prec_idx += 1
                
                # If all recall levels reached, move onto next image
                if success_count == 9:
                    all_neighbours_found = True
                    break
            else:
                continue
        
        assert all_neighbours_found == True, "Didn't find all recall levels: {}".format(index)
        interpolated_prec = interpolate_precision(precisions)
        average_precisions[index] = interpolated_prec.mean()

    return (acc_1.mean(), acc_10.mean(), average_precisions.mean())

####################################################
####################################################
# Pop an element from an ndarray, and similar to 
# list.pop() but can specify index
# Returns element, new_array
def pop(data, index=-1):
    shape = data.shape
    if len(shape)==2:
        new_array = np.empty(shape=(shape[0]-1,shape[1]))
        element = np.empty(shape=(1,shape[1]))
        element[0] = data[index]
    else:
        new_array = np.empty(shape=(shape[0]-1))
        element = data[index]
    
    if (index == -1) or (index == shape[0]-1):
        new_array = data[:index]
    elif index == 0:
        new_array = data[index+1:]
    else:
        new_array = np.concatenate((data[:index], data[index+1:]))
        
    return element, new_array

####################################################
####################################################
# Calculate interpolated precision from precision_recall curve
def interpolate_precision(prec_recall_curve):
    interpolated_precision = np.empty(shape=(prec_recall_curve.shape))
    
    for idx, _ in enumerate(prec_recall_curve):
        interpolated_precision[idx] = np.max(prec_recall_curve[idx:])
    
    return interpolated_precision


###############################################################
###############################################################
# Returns classifier set to the correct metric
def get_classifier(num_neighbors, metric, arg=None):
    if metric == 'mahalonobis':
        knn = KNeighborsClassifier(n_neighbors=num_neighbors, algorithm='ball_tree', metric=distance, metric_params={'VI': arg})
        
    elif metric == 'KL':
#         knn = KNeighborsClassifier(n_neighbors=num_neighbors, metric=lambda a,b:KL_div(a,b) )
        knn = KNeighborsClassifier(n_neighbors=num_neighbors, algorithm='ball_tree', metric=KL_div)
        
    elif metric == 'JS':
        knn = KNeighborsClassifier(n_neighbors=num_neighbors, algorithm='ball_tree', metric=JS_div )
        
    elif metric == 'chi_square':
        knn = KNeighborsClassifier(n_neighbors=num_neighbors, algorithm='ball_tree', metric=chi_square )
        
    elif metric == 'intersection':
        knn = KNeighborsClassifier(n_neighbors=num_neighbors, algorithm='ball_tree', metric=intersection )
        
    elif metric == 'cross_correlation':
        knn = KNeighborsClassifier(n_neighbors=num_neighbors, algorithm='ball_tree', metric=cross_correlation )
        
    elif metric == 'cosine_similarity':
        knn = KNeighborsClassifier(n_neighbors=num_neighbors, algorithm='ball_tree', metric=cosine_similarity )
        
    elif metric == 'earth_mover':
        knn = KNeighborsClassifier(n_neighbors=num_neighbors, algorithm='ball_tree', metric=earth_mover )
        
    elif metric == 'euc_sqr':
        knn = KNeighborsClassifier(n_neighbors=num_neighbors, algorithm='ball_tree', metric=euc_sqr )
        
    elif metric == 'qf_distance':
        knn = KNeighborsClassifier(n_neighbors=num_neighbors, algorithm='ball_tree', 
                                   metric=qf_distance, metric_params={'A': arg} )
    elif metric == 'qc_distance':
        knn = KNeighborsClassifier(n_neighbors=num_neighbors, algorithm='ball_tree', 
                                   metric=qc_distance, metric_params={'A': arg} )
    else:
        knn = KNeighborsClassifier(n_neighbors=num_neighbors, metric=metric)
    return knn 

####################################################
###############     Projections      ############### 
####################################################

####################################################
####################################################
# Get linear transform used for mahalnobis
def get_lin_transform(x_data):
    inv_cov_matrix = np.linalg.inv(np.cov(x_data.T))
    eig_vals, eig_vecs = np.linalg.eig(inv_cov_matrix)
    eig_vals = np.real(eig_vals)
    eig_vecs = np.real(eig_vecs)
    # Sort eigenstuff from greatest to smallest
    idx = eig_vals.argsort()[::-1]
    eig_vals = eig_vals[idx]   # <-- This sorts the list in place.
    eig_vecs = eig_vecs[:, idx]
    eig_vals_sqrt = np.diag(np.sqrt(eig_vals))

    return np.matmul(eig_vals_sqrt, eig_vecs.T)

###############################################################
###############################################################
# Get NCA transformed data
def nca(x_train, y_train, x_test, n_c):
    nca = NeighborhoodComponentsAnalysis(n_c)
    nca.fit(x_train, y_train)
    return nca.transform(x_test)

###############################################################
###############################################################
# Get LMNN transformed data
def lmnn(x_train, y_train, x_test):
    lmnn = LMNN(max_iter=50, k=9, verbose=True)
    print("It is")
    lmnn.fit(x_train, y_train)
    print("done")
    return lmnn.transform(x_test)
    
####################################################
####################################################
# Transforms each element of x_data to a histogram
# containing the specified numbers of bins
def project_to_histogram(x_data, bins=20):
    new_array = np.empty(shape=(x_data.shape[0], bins))
    for idx, element in enumerate(x_data):
        new_array[idx], _ = np.histogram(element, bins=bins)
    return new_array

###############################################################
###############################################################
# Fisherfaces - combines PCA + LDA, returns transformed 
# training and testing set

def fisherface(x_train, y_train, x_test, M_pca, M_lda):
    proj, comp, mean, centered_data = pca(x_train, M_pca)
    x_test = transform(x_test, comp, mean)
    
    lda = LDA(n_components=M_lda)
    x_train = lda.fit_transform(proj, y_train)
    x_test = lda.transform(x_test)
    
    return x_train, x_test

###############################################################
###############################################################
# Perform low-dimensional PCA
def pca(X, n_pc):
    n_samples, n_features = X.shape
    mean = np.mean(X, axis=0)
    centered_data = X-mean
    U, S, V = np.linalg.svd(centered_data)
    components = V[:n_pc]
    projected = U[:,:n_pc]*S[:n_pc]
    return projected, components, mean, centered_data

###############################################################
###############################################################
# Transform X to PCA subspace
def transform(X, components, mean):
    X = X - mean
    X_trans = np.dot(X, components.T)
    return X_trans

####################################################
###############       Metrics        ############### 
####################################################
    
###############################################################
###############################################################
# Calculate cosine similarity of two vectors
def cosine_similarity(X, Y):
    top = np.dot(X.T, Y)
    bottom = np.linalg.norm(X) * np.linalg.norm(Y)
#     return top / bottom
    return 1.0 - abs(top / bottom)

###############################################################
###############################################################
# Calculate cross correlation of two vectors
def cross_correlation(X, Y):
    return 1.0 - np.multiply(X,Y).sum()

###############################################################
###############################################################
# Calculate KL divergence of two vectors
def KL_div(X, Y):
    func = np.vectorize(lambda x,y: x * np.log((x+1e-6)/(y+1e-6)))
    array = func(X,Y)
    return array.sum()
#     return scipy.stats.entropy(X,Y)
    

###############################################################
###############################################################
# Calculate JS divergence of two vectors
def JS_div(X, Y):
    M = (X+Y) / 2
    return np.sqrt((KL_div(X,M) + KL_div(Y,M)) * 0.5)

###############################################################
###############################################################
# Calculate ChiSquare of two vectors
def chi_square(X, Y):
    func = np.vectorize(lambda x,y: pow(x-y,2) / (x+y+1e-6) )
    array = func(X,Y)
    
    return np.sqrt(array.sum() * 0.5)

###############################################################
###############################################################
# Calculate squared euclidean distance
def euc_sqr(X, Y):
    return scipy.spatial.distance.euclidean(X,Y) ** 2

####################################################
###############      Hist Metrics    ############### 
####################################################

###############################################################
###############################################################
# Calculate earth mover's distance of two vectors
def earth_mover(X, Y):
    return scipy.stats.wasserstein_distance(X,Y)

###############################################################
###############################################################
# Calculate quadratic form histogram distance
def qf_distance(X, Y, A):
    diff = X-Y
    left = np.dot(diff.T, A)
    full = np.dot(left, diff)
    return np.sqrt(full)

###############################################################
###############################################################
# Calculate quadratic chi histogram distance
def qc_distance(X, Y, A, m=1):
    array = np.empty(shape=(A.shape))
    vec_sum = (X+Y)
    for i, x_i in enumerate(X):
        y_i = Y[i]
        diff_i = x_i - y_i
        bottom_left = (np.multiply(vec_sum, A[:][i])).sum() ** m
        
        for j, x_j in enumerate(X):
            y_j = Y[j]
            diff_j = x_j - y_j 
            top = np.dot(diff_i, diff_j) * A[i][j]
            bottom_right = (np.multiply(vec_sum, A[:][j])).sum() ** m
            array[i][j] = top / (bottom_left*bottom_right)
    return np.sqrt(array.sum())

###############################################################
###############################################################
# Calculate intersection of two vectors
def intersection(X, Y):
    min_sum = np.minimum(X,Y).sum()
    X_sum = X.sum()
    Y_sum = Y.sum()
    
    return 1.0 - (min_sum / X_sum + min_sum / Y_sum) * 0.5
#     return (min_sum / X_sum + min_sum / Y_sum) * 0.5


####################################################
###############          Q2          ############### 
####################################################


####################################################
####################################################
# Function that computes and returns the centroids of a
# set of clusters especially for AgglomerativeClustering
def get_agglomClustering_centroids(data, labels, n_clusters):
    n_samples, n_features = np.array(data).shape
    clusters = [np.array([], dtype=float) for _ in range(n_clusters)]
    # 1. divide data into clusters
    for i,l in enumerate(labels):
        p = data[i]
        cluster_pop = len(clusters[l])
        clusters[l] = np.append(clusters[l], np.array(p)).reshape((cluster_pop+1, n_features))
    
    
    # 2. compute clusters means
    centroids = np.array([np.mean(c, axis=0) for c in clusters], dtype=float)
    return centroids

####################################################
####################################################
# Function that computes the inertia  of a set of clusters
# especially for AgglomerativeClustering
# Inertia: sum of distances of points to the mean of the cluster
def get_agglomClustering_inertia(data, labels, n_clusters):
    centroids = get_agglomClustering_centroids(data, labels, n_clusters)
    # compute inertia
    inertia = 0.0
    for i,l in enumerate(labels):
        inertia += np.linalg.norm(np.subtract(data[i],centroids[l]))
    return inertia

####################################################
####################################################
# Function that assigns datapoints to clusters,
# given centroids
# Returns   list of n np.arrays with datapoints (n is n_clusters)
#           list of labels of datapoints
#           inertia value of the generated clusters
def assign_datapoints(data, centroids, distance='euclidean'):
    n_samples, n_features = np.array(data).shape
    inertia = 0.0
    clusters = [np.array([], dtype=float) for _ in range(len(centroids))]
    labels = np.empty(shape=(n_samples), dtype=int)
    for i,p in enumerate(data):
        if distance=='euclidean':
            distances = [np.linalg.norm(np.subtract(p,c)) for c in centroids]
        else:
            raise NotImplementedError
        assigned_idx = np.argmin(distances)
        cluster_pop = len(clusters[assigned_idx])
        clusters[assigned_idx] = np.append(clusters[assigned_idx], np.array(p)).reshape((cluster_pop+1, n_features))
        labels[i] = assigned_idx
        inertia += np.amin(distances)
    return clusters, labels, inertia

####################################################
####################################################
# Function that performs k_means
# Returns   centroid positions
#           input set labels
#           final inertia value
#           (if return_n_iter) --> number of iterations for achieving best result 
def k_means(X, n_clusters, init='random', n_init=10, 
        max_iter=300, verbose=False, tol=1e-4,
        random_state=None, return_n_iter=False):

    n_samples, n_features = X.shape
    centroids = np.zeros(shape=(n_clusters, n_features), dtype=float)
    labels = np.empty(shape=(n_samples,))

    # 1. Choose initial centroids
    if init=='random': # assign to (n_clusters) randomly chosen datapoints 
        temp_copy = np.array(X,dtype=float)
        np.random.shuffle(temp_copy)
        centroids = temp_copy[:n_clusters]
    elif init=='kmeans++':
        raise NotImplementedError
    else:
        raise ValueError

    # 2. Optimize centroids
    n_iter = 0
    labels = [0] #empty init
    old_labels = [1] #empty init
    inertia = 1

    clusters, labels, inertia = assign_datapoints(X, centroids)
    while(  n_iter < max_iter and inertia > tol and not np.array_equal(labels,old_labels)):
        old_labels = labels
        n_iter += 1
        # 2b. Update centroids
        centroids = np.array([np.mean(c, axis=0) for c in clusters], dtype=float)
        # 2a. Assign datapoints to centroids
        clusters, labels, inertia = assign_datapoints(X, centroids)
        if verbose:
            print("Iteration",n_iter,"---", "Inertia",round(inertia,3),sep="     \t", end="\r")
    if verbose:
        print("\n=============================")
    return centroids, labels, inertia, n_iter

####################################################
####################################################
# Function that assigns class labels in range(n_clusters)
# to the clusters based on the true labels of the data in 
# each cluster. The algorithm used is the Hungarian Algorithm
# Returns   ordered np array of lables
# ======>> !!!TODO!!! implement from scratch Hungarian Algorithm <<======
def assign_labels_to_clusters(assigned_labels, true_labels, n_clusters, class_offset=0, algorithm="hungarian"):
    # count class instances in each cluster
    class_count = np.zeros(shape=(n_clusters,n_clusters), dtype=int)
    for idx, l in enumerate(assigned_labels):
        class_count[l][true_labels[idx]-class_offset-1] += 1
    # turn maximisation problem into minimization
    min_class_count = np.full((n_clusters, n_clusters), np.amax(class_count)) - class_count
    # solve using Hungarian algorithm
    if algorithm=="hungarian":
        _, cluster_labels = linear_sum_assignment(min_class_count) 
    else:   
        raise NotImplementedError
    return cluster_labels+class_offset+1

####################################################
####################################################
# Given the labels assigned to clusters, compute
# the classification accuracy of each cluster
# and overall accuracy
def get_labelledCluster_predictionAccuracy(assigned_labels, true_labels,cluster_labels,
                                           n_clusters,class_offset):
    # count correct predicions for each cluster
    # and count population of cluster
    total_correct = 0
    correct_count = np.zeros(shape=(n_clusters), dtype=int)
    cluster_population = np.zeros(shape=(n_clusters), dtype=int)
    for idx, l in enumerate(assigned_labels):
        # check if element is in the cluster with the same label
        if cluster_labels[l] == true_labels[idx]:
            correct_count[l] += 1
            total_correct += 1
        # increase population counter
        cluster_population[l] += 1
        
    # scale accuracy for population of each cluster
    accuracies = [correct_count[c]/cluster_population[c] for c in range(n_clusters)]
    overall_accuracy = total_correct/len(assigned_labels)
    return accuracies, overall_accuracy

####################################################
####################################################
# Generate fisher vectors representation based on GMM
# parameters
def generate_fisher_vectors(X, probabilities, means, variances, weights):
    n_clusters, n_features = np.array(means).shape
    n_samples, _ = np.array(X).shape

    out = []

    for i,img in enumerate(X):
        fv = []
        for k in range(n_clusters):
            # compute v_k
            temp1 = np.divide(np.subtract(img, means[k]), variances[k])
            temp2 = temp1 * (probabilities[i][k])
            v_k = temp2 / math.sqrt(weights[k])
            # compute u_k
            temp3 = pow(temp1-1, 2)
            temp4 = temp3 * (probabilities[i][k])
            u_k = temp4 / math.sqrt(2*weights[k]) 
            # append v_k and u_k to fisher vector
            fv.append(v_k)
            fv.append(u_k)
        # flattend and normalize and sqrtfisher vector
        fv_normalized = np.array(fv).flatten().reshape(1, -1)
        fv_normalized = preprocessing.normalize(fv_normalized).reshape(2*n_clusters*n_features)
        fv_normalized = np.nan_to_num(np.sqrt(fv_normalized))
        # add fisher vector to output list
        out.append(fv_normalized)

    #return out.reshape(n_samples,n_clusters*n_features)
    return out