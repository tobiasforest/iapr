from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import cv2
import matplotlib.pyplot as plt

def build_gabor_filters():
    """
    Builds a set of Gabor filters.

    Args:
        None

    Returns:
        filters (list): A list of Gabor filters.

    Raises:
        None

    Examples:
        >>> build_gabor_filters()
        [filter1, filter2, filter3]
    """
    filters = []
    ksize = 31
    # Define the number of orientations and scales for the Gabor filters
    for theta in np.arange(0, np.pi, np.pi / 16):
        for sigma in np.arange(0.5, 2.5, 0.5):
            kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
            kern /= 1.5*kern.sum()
            filters.append(kern)
    return filters

def apply_gabor_filters(image, filters):
    """
    Applies a set of Gabor filters to an image.

    Args:
        image (numpy.ndarray): The input image.
        filters (list): The list of Gabor filters.

    Returns:
        responses (list): The list of responses of the Gabor filters.
        
    Raises:
        None

    Examples:
        >>> apply_gabor_filters(image, filters)
        [response1, response2, response3]
    """
    responses = []
    for kern in filters:
        filtered_image = cv2.filter2D(image, cv2.CV_8UC3, kern)
        responses.append(filtered_image)
    return responses

def color_histogram(image, bins=(8, 8, 8)):
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def gabor_histogram(image, filters, bins=10):
    responses = apply_gabor_filters(image, filters)
    abs_responses = np.concatenate([np.abs(response).flatten() for response in responses])
    hist = np.histogram(abs_responses, bins=bins, range=(0, 256))[0]
    return hist.flatten()

def color_range(image):
    ranges = [np.ptp(image[:,:,i]) for i in range(image.shape[2])]
    return np.array(ranges)

def extract_features(image):
    """
    Extracts the features from an image.

    Args:
        image (numpy.ndarray): The input image.

    Returns:
        bool: True if the contours are twins, False otherwise.

    Raises:
        None

    Examples:
        >>> twin_contours(contour1, contour2)
        True
        >>> twin_contours(contour1, contour3)
        False
    """
    filters = build_gabor_filters()
    gabor_features = gabor_histogram(image, filters)
    color_features = color_histogram(image)
    color_ranges = color_range(image)
    features = np.concatenate((gabor_features, color_features, color_ranges))
    return features

def perform_pca(data, n_components=10):
    
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Create PCA instance
    pca = PCA(n_components=n_components)
    
    # Perform PCA
    pca_result = pca.fit_transform(data)
    
    return pca_result

def extract_features_from_pieces(pieces):
    return np.array([extract_features(piece) for piece in pieces])
    
def plot_features(features):
    plt.figure(figsize=(20, 10))
    # plot with ordered numbers
    plt.plot(range(len(features)), features)
    plt.show()

def plot_features_from_pieces(pieces):
    features = extract_features_from_pieces(pieces)
    plot_features(features)
