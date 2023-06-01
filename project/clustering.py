from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
import cv2
from sklearn.preprocessing import MinMaxScaler
import segmentation as seg
import feature as feat

def clusters_check(features, labels):
    labels = check_outlier_cluster(labels, max_num_outliers=3)
    labels = combine_clusters(labels, acceptable_sizes=(9, 12, 16))
    labels = separate_mixed_cluster(features, labels, acceptable_sizes=(9, 12, 16))
    return labels

def outliers_k_means(labels, num_outliers_max=3):
    cluster_counts = np.bincount(labels)
    if cluster_counts[np.argmin(cluster_counts)] <= num_outliers_max: 
        outlier_cluster = np.argmin(cluster_counts)
        labels[labels == outlier_cluster] = -1
    else:
        outlier_cluster = None
    return labels


def remove_false_outliers(labels, max_num_outliers=3):
    outlier_indices = np.where(labels == -1)[0]
    num_outliers = outlier_indices.shape[0]
    if num_outliers > max_num_outliers:
        new_cluster_index = np.max(labels) + 1
        labels[outlier_indices] = new_cluster_index
    return labels


def detect_outlier_cluster(labels, acceptable_sizes=(9,12,16), max_num_outliers=3):
    unique_labels = np.unique(labels)
    if -1 not in unique_labels:
        cluster_sizes=[]
        for label in unique_labels:
            cluster_pts_idx = np.where(labels == label)[0]
            cluster_size = cluster_pts_idx.shape[0]
            cluster_sizes.append(cluster_size)
        count_acceptable_sizes = sum(size in acceptable_sizes for size in cluster_sizes)
        if count_acceptable_sizes == len(cluster_sizes) - 1:
            outlier_cluster_index=[index for index, size in enumerate(cluster_sizes) if size not in acceptable_sizes]
            outlier_cluster_label = unique_labels[outlier_cluster_index]
            labels[labels==outlier_cluster_label]=-1
        else:
            print("There are multiple clusters that don't have the acceptable size.")
    return labels

def check_outlier_cluster(labels, max_num_outliers=3):
    remove_false_outliers(labels)
    detect_outlier_cluster(labels)
    return labels

def combine_clusters(labels, acceptable_sizes=(9, 12, 16)):
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels!=-1]
    for i, label in enumerate(unique_labels):
        cluster_pts_idx = np.where(labels == label)[0]

        cluster_size = cluster_pts_idx.shape[0]
        if cluster_size < min(acceptable_sizes):
            for j, label2 in enumerate(np.unique(labels)):
                if j == i:
                    continue
                else:
                    other_cluster_pts_idx = np.where(labels == label2)[0]
                    other_cluster_size = other_cluster_pts_idx.shape[0]
                    combined_size = cluster_size + other_cluster_size
                    if combined_size in acceptable_sizes:
                        labels[other_cluster_pts_idx] = label
    
    return labels

def separate_mixed_cluster(features, labels, acceptable_sizes=(9, 12, 16)):
    combinations_sizes = []
    for i in acceptable_sizes:
        for j in acceptable_sizes:
            combinations_sizes.append(i+j)
    
    combinations_sizes = np.unique(combinations_sizes)
    unique_labels = np.unique(labels)
    
    # Remove the outlier cluster
    unique_labels=unique_labels[unique_labels!=-1]
    
    # Iterate over the clusters
    for i, label in enumerate(np.unique(labels)):
        # Check if cluster size is in combinations_sizes
        cluster_pts_idx = np.where(labels == label)[0]
        cluster_size = cluster_pts_idx.shape[0]
        
        if cluster_size in combinations_sizes:
            cluster_pts = features[cluster_pts_idx]
            
            # Use kmeans to separate the cluster into 2 clusters
            kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
            labels2 = kmeans.fit_predict(cluster_pts)
            
            # Modify labels 2 so that we create a new cluster label
            labels2=labels2+np.max(labels)+1
            
            # Update labels 
            labels[cluster_pts_idx]=labels2
    return labels

def get_PCA_features(features, n_components=3):
    pca = KernelPCA(n_components=n_components,kernel="rbf")
    features_PCA = pca.fit_transform(features)
    return features_PCA

def cluster_features(features, max_clusters=6):
    kmeans_models = []
    inertia_values = []
    silhouette_scores = []
    last_inertia = None
    inertia_changes = []

    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
        cluster_labels = kmeans.predict(features)
        silhouette_avg = silhouette_score(features, cluster_labels)

        kmeans_models.append(kmeans)
        inertia_values.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_avg)

        if last_inertia is not None:
            inertia_changes.append((last_inertia - kmeans.inertia_) / last_inertia)
        last_inertia = kmeans.inertia_

    return kmeans_models, inertia_values, silhouette_scores, inertia_changes

def plot_clustering_scores(inertia_values, silhouette_scores):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(2, len(inertia_values) + 2), inertia_values, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Inertia of k-Means versus number of clusters')

    plt.subplot(1, 2, 2)
    plt.plot(range(2, len(silhouette_scores) + 2), silhouette_scores, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')
    plt.title('Silhouette score of k-Means versus number of clusters')

    plt.tight_layout()
    plt.show()

def plot_clustered_pieces(pieces, labels):
    unique_labels = set(labels)

    # count the number of pieces in each cluster, computed only once
    piece_counts = {curr_label: len([label for label in labels if label == curr_label]) for curr_label in unique_labels}

    # get the maximum number of pieces in a cluster
    max_pieces = max(piece_counts.values())

    for label in unique_labels:
        indices = [index for index, curr_label in enumerate(labels) if curr_label == label]
        cluster_pieces = [pieces[index] for index in indices]

        plt.figure(figsize=(15, 5))

        # normalize image size using a list comprehension
        normalized_cluster_pieces = [cv2.resize(piece, (50, 50)) for piece in cluster_pieces]

        for i in range(max_pieces):
            plt.subplot(1, max_pieces, i+1)

            # if the cluster has fewer pieces, we fill the remaining subplots with blank images
            if i < len(normalized_cluster_pieces):
                plt.imshow(cv2.cvtColor(normalized_cluster_pieces[i], cv2.COLOR_BGR2RGB))
            else:
                pass

            plt.axis('off')

        plt.suptitle(f'Cluster {label}', fontsize=20)
        plt.tight_layout()
        plt.show()

def find_optimal_clusters(inertia_values, silhouette_scores):
    # calculate the 1st and 2nd derivative of the inertia and silhouette scores
    inertia_first_derivative = np.diff(inertia_values, prepend=0)
    inertia_second_derivative = np.diff(inertia_first_derivative, prepend=0)

    silhouette_first_derivative = np.diff(silhouette_scores, prepend=0)
    silhouette_second_derivative = np.diff(silhouette_first_derivative, prepend=0)

    # calculate the 'elbow point' using the 2nd derivative (i.e., the point of maximum curvature)
    elbow_point = np.argmax(inertia_second_derivative) + 2  # +2 because the range starts from 2

    # scale values between 0 and 1
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(np.array([inertia_values,
                                                   silhouette_scores,
                                                   inertia_first_derivative,
                                                   silhouette_first_derivative,
                                                   abs(np.arange(len(inertia_values)) - elbow_point)]).T)
    
    scaled_values = np.array([inertia_values,
                                silhouette_scores,
                                inertia_first_derivative,
                                silhouette_first_derivative,
                                abs(np.arange(len(inertia_values)) - elbow_point)]).T
    norm_scaled_values = np.linalg.norm(scaled_values, axis=1)
    
    scaled_values = scaled_values / norm_scaled_values[:, None]
    scores = []
    for i in range(len(scaled_values)):
        try:
            # here we consider five factors:
            # 1) inertia: we want to minimize this (hence the negative sign)
            # 2) silhouette score: we want to maximize this
            # 3) inertia change: we want to minimize this (hence the negative sign)
            # 4) silhouette change: we want to maximize this
            # 5) distance to elbow point: we want to minimize this (hence the negative sign)
            # the factors are combined as a weighted sum, adjust the weights as needed
            score = -scaled_values[i, 0]  + scaled_values[i, 1] 
            # We discarded the rest for better results
            scores.append(score)
        except ZeroDivisionError:
            continue

    optimal_clusters = scores.index(max(scores)) + 2  # +2 because the range starts from 2
    return optimal_clusters

def get_labels_pieces(features):
    
    features = features / np.linalg.norm(features, axis=1)[:, None]
    features = get_PCA_features(features, n_components=4)
    kmeans_models, inertia_values, silhouette_scores, inertia_changes = cluster_features(features)
    optimal_clusters = find_optimal_clusters(inertia_values, silhouette_scores)
    
    kmeans = kmeans_models[optimal_clusters - 2]  # -2 because the range starts from 2
    labels = kmeans.predict(features)
    
    labels = outliers_k_means(labels)
    labels = clusters_check(features, labels)
    
    return labels

def get_cluster_list(image):
    pieces = seg.extract_pieces_from_image(image)
    features = feat.extract_features_from_pieces(pieces)
    labels = get_labels_pieces(features)
    unique_labels = set(labels)
    print(unique_labels)
    cluster_list = []
    outlier_list = []
    for label in unique_labels:
        if label == -1:
            outlier_list = [pieces[index] for index, curr_label in enumerate(labels) if curr_label == label]
        else:
            indices = [index for index, curr_label in enumerate(labels) if curr_label == label]
            cluster_pieces = [pieces[index] for index in indices]
            cluster_list.append(cluster_pieces)
    print(len(cluster_list))
    cluster_list.append(outlier_list)
    return cluster_list
