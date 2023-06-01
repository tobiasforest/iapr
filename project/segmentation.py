import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage import filters

def plot_color_histograms(image_path: str, title: str = ''):
    """
    Plots the color histograms for the input image.

    Args:
        image_path (str): The path to the input image.

    Returns:
        None

    Raises:
        None

    Examples:
        >>> plot_color_histograms('train_00.png', 'Train Image 00')
    """
    
    image = cv2.imread(image_path)
    # Convert the image from BGR to RGB format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Calculate the histograms for each color channel
    hist_red = cv2.calcHist([image_rgb], [0], None, [256], [0, 256])
    hist_green = cv2.calcHist([image_rgb], [1], None, [256], [0, 256])
    hist_blue = cv2.calcHist([image_rgb], [2], None, [256], [0, 256])

    # Plot the histograms
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.plot(hist_red, color='red')
    plt.title('Red Histogram')
    plt.xlim([0, 256])
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')

    plt.subplot(1, 3, 2)
    plt.plot(hist_green, color='green')
    plt.title('Green Histogram')
    plt.xlim([0, 256])
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')

    plt.subplot(1, 3, 3)
    plt.plot(hist_blue, color='blue')
    plt.title('Blue Histogram')
    plt.xlim([0, 256])
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    
def twin_contours(contour1, contour2):
    """
    Checks if two contours are twins.

    Args:
        contour1 (numpy.ndarray): The first contour.
        contour2 (numpy.ndarray): The second contour.

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
    
    # Get the bounding rectangles for the two contours
    x1, y1, w1, h1 = cv2.boundingRect(contour1)
    x2, y2, w2, h2 = cv2.boundingRect(contour2)

    centroid1 = (x1 + w1 // 2, y1 + h1 // 2) # (x, y) coordinates of the centroid
    centroid2 = (x2 + w2 // 2, y2 + h2 // 2)

    distance = np.linalg.norm(np.array(centroid1) - np.array(centroid2))
    length_side_square = 128.0
    return distance < length_side_square / 2.0

def combine_contours(contours_red, contours_green, contours_blue):
    """
    Combines the contours from the three color channels.

    Args:
        contours_red (list): The list of contours from the red channel.
        contours_green (list): The list of contours from the green channel.
        contours_blue (list): The list of contours from the blue channel.

    Returns:
        list: The list of all contours.

    Raises:
        None

    Examples:
        >>> combine_contours(contours_red, contours_green, contours_blue)
        [(contour1, 'red'), (contour2, 'green'), (contour3, 'blue')]
    """
    all_contours = [(contour, 'red') for contour in contours_red]
    all_contours += [(contour, 'green') for contour in contours_green]
    all_contours += [(contour, 'blue') for contour in contours_blue]

    return all_contours

def get_final_contours(all_contours):
    """
    Gets the final contours from the list of all contours.

    Args:
        all_contours (list): The list of all contours.

    Returns:
        list: The list of final contours.

    Raises:
        None

    Examples:
        >>> get_final_contours(all_contours)
        [contour1, contour2, contour3]
    """
    final_contours = []
    for i in range(len(all_contours)):
        is_twin = False
        for j in range(i + 1, len(all_contours)):
            if twin_contours(all_contours[i][0], all_contours[j][0]):
                is_twin = True
                if cv2.contourArea(all_contours[i][0]) < cv2.contourArea(all_contours[j][0]):
                    all_contours[i] = all_contours[j]
                break
        if not is_twin:
            final_contours.append(all_contours[i])

    return final_contours

def get_contours_channel(image, threshold=0.02, channel=0):
    """
    Gets the contours from the specified channel of the image.

    Args:
        image (numpy.ndarray): The input image.
        threshold (float): The threshold value for the Sobel filter.
        channel (int): The channel to use for the contours.

    Returns:
        list: The list of contours.

    Raises:
        None

    Examples:
           get_contours_channel(image, threshold=0.02, channel=0)
        [contour1, contour2, contour3]
    """
    image_sobel = filters.sobel(image[..., channel])

    markers = np.zeros_like(image_sobel)
    markers[image_sobel > threshold * image_sobel.max()] = 255
    
    mask = markers.astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 128 x 128 pixels ~ 16000
    min_area_threshold = 10000
    max_area_threshold = 20000

    # Iterate over the contours and filter based on area and aspect ratio
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        _, _, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h

        # Adjust the area and aspect ratio thresholds as per your requirements
        if area > min_area_threshold and area < max_area_threshold and aspect_ratio > 0.8 and aspect_ratio < 1.2:
            filtered_contours.append(contour)
            # Draw the contour on the image
    return filtered_contours

def get_convex_contours(contours):
    """
    Gets the convex contours from the list of contours.

    Args:
        contours (list): The list of contours.

    Returns:
        list: The list of convex contours.

    Raises:
        None

    Examples:
        >>> get_convex_contours(contours)
        [convex_contour1, convex_contour2, convex_contour3]
    """
    convex_contours = []
    for contour in contours:
        convex_contour = cv2.convexHull(contour)
        convex_contours.append(convex_contour)
    return convex_contours

def segmentate_pieces(image, threshold=0.02):
    """
    Segments the pieces from the input image.

    Args:
        image (numpy.ndarray): The input image.
        threshold (float): The threshold value for the Sobel filter.
        
    Returns:
        list: The list of contours of the pieces after complete segmentation.

    Raises:
        None

    Examples:
        >>> segmentate_pieces(image, threshold=0.02)
        [contour1, contour2, contour3]
    """
    red_contours = get_contours_channel(image, threshold, 2)
    green_contours = get_contours_channel(image, threshold, 1)
    blue_contours = get_contours_channel(image, threshold, 0)

    final_contours = get_final_contours(red_contours + green_contours + blue_contours)

    convex_contours = get_convex_contours(final_contours)

    image_with_contours = image.copy()
    for contour in convex_contours:
        cv2.drawContours(image_with_contours, [contour], 0, (0, 0, 255), 5)

    return convex_contours

def get_mask(image):
    correct_contours = segmentate_pieces(image)
    # Create an empty black image of the same size as the input image
    mask = np.zeros_like(image)
    # Draw contours on the black image
    cv2.drawContours(mask, correct_contours, -1, (255), thickness=-1)
    
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    return mask_gray


def get_rotated_crop(image, contour):
    # Get the minimum enclosing rectangle
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # Compute the rotation angle
    angle = rect[2]

    # Correct the angle
    if angle < -45:
        angle += 90

    # Get the center of the rectangle
    center = (rect[0][0], rect[0][1])

    # Generate the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1)

    # Perform the rotation on the original image
    rotated_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # Rotate the contour in the same way
    rotated_contour = cv2.transform(contour, M)

    # Recalculate the bounding box based on the rotated contour
    x, y, w, h = cv2.boundingRect(rotated_contour)

    # Calculate the center coordinates of the rotated bounding box
    center_x = x + w // 2
    center_y = y + h // 2

    # Calculate the crop boundaries based on the desired size (128, 128)
    crop_size = 128
    x_min = max(0, center_x - crop_size // 2)
    y_min = max(0, center_y - crop_size // 2)
    x_max = min(image.shape[1], x_min + crop_size)
    y_max = min(image.shape[0], y_min + crop_size)

    # Crop the rotated image
    cropped_image = rotated_image[y_min:y_max, x_min:x_max]

    return cropped_image


def extract_pieces_from_image(image_path):
    image = cv2.imread(image_path)
    contours = segmentate_pieces(image)
    pieces = [get_rotated_crop(image, contour) for contour in contours]
    return pieces

def plot_pieces(pieces):
    num_pieces = len(pieces)

    # Create a large and multi-column figure
    plt.figure(figsize=(20, 10))
    columns = 5
    for i, piece in enumerate(pieces):
        plt.subplot(num_pieces / columns + 1, columns, i + 1)
        plt.imshow(cv2.cvtColor(piece, cv2.COLOR_BGR2RGB))
        plt.axis('off')
    plt.show()

