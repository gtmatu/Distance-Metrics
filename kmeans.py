from cw2_library import k_means, assign_labels_to_clusters, generate_fisher_vectors
import numpy as np
from scipy.special import softmax
from sklearn.mixture import GaussianMixture

NCLASSES_TESTSET = 20 # Number of different classes in the test set
CLASS_OFFSET=52-NCLASSES_TESTSET
####################################################
####################################################
# Class for KMeans clustering
class KMeans:

    def __init__(self, n_clusters=8, init="random", n_init=10,
                        max_iter=300, verbose=False, random_state=None):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X, true_labels=None):
        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = k_means(X, n_clusters=self.n_clusters, init=self.init,
                                                                                    n_init=self.n_init, max_iter=self.max_iter,
                                                                                    verbose=self.verbose, random_state=self.random_state,
                                                                                     return_n_iter=True)
        if true_labels:
            self.cluster_labels = assign_labels_to_clusters(assigned_labels=self.labels_,
                                                true_labels=true_labels,
                                                n_clusters=self.n_clusters,
                                                class_offset=CLASS_OFFSET)
        return self
    
    def predict(self, X):
        n_samples, n_features = X.shape
        if n_features != self.cluster_centers_.shape[1]:
            raise ValueError
        predictions = np.empty(shape=(n_samples), dtype=int)
        for i, img in enumerate(X):
            predictions[i] = np.argmax([np.linalg.norm(np.subtract(img, centroid)) for centroid in self.cluster_centers_])
        return predictions

    def project(self, X, training_set=None, rule=None):
        n_samples, _ = X.shape
        n_features = self.n_clusters
        
        if rule=="distance_to_center":
            projected = np.empty(shape=(n_samples, self.n_clusters))
            for i,sample in enumerate(X):
                for j,center in enumerate(self.cluster_centers_):
                    projected[i][j] = np.linalg.norm(np.subtract(sample, center))
        
        elif rule=="softmax_inverse_distance":
            projected = self.project(X, rule="distance_to_center")
            projected_inverse = np.divide(np.full(shape=projected.shape, fill_value=1, dtype=float), projected)
            projected = softmax(np.nan_to_num(projected_inverse), axis=1)
        
        elif rule=="fisher_vector_gmm":
            if training_set is None:
                raise ValueError 

            # associate points to clusters
            clusters = [[] for _ in range(self.n_clusters)]
            for i,label in enumerate(self.labels_):
                clusters[label].append(training_set[i])
            # find variances for each cluster (kx2576)
            precisions = []
            for cluster in clusters:
                cov_matrix = np.absolute(np.cov(np.array(cluster).T))
                
                inverse_var_vector = np.array([1.0/(np.absolute(cov_matrix[i][i])+1e-6) for i in range(cov_matrix.shape[0])])
                precisions.append(np.nan_to_num(inverse_var_vector)+1e-6)
            # print(np.array(precisions).shape)
                # print(sum(sum([a<0 for a in precisions])))
            # calculate weights --> (imgs in cluster/total images), shape=k
            weigths = [len(cluster)/len(training_set) for cluster in clusters]
            # Calculate GMM representation from cluster centers
            gmm = GaussianMixture(  n_components=self.n_clusters,
                                    covariance_type="diag",
                                    means_init=self.cluster_centers_,
                                    precisions_init = precisions,
                                    weights_init = weigths)
            # fit gmm with training data
            gmm = gmm.fit(training_set)
            # predict_proba on testing data (gamma)
            gamma = gmm.predict_proba(X)
            # calculate fisher vectors
            projected = generate_fisher_vectors(X=X, 
                                                probabilities=gamma, 
                                                means=gmm.means_,
                                                variances=gmm.covariances_, 
                                                weights=gmm.weights_)

        else:
            raise NotImplementedError

        return projected