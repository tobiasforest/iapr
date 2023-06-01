import os
from PIL import Image
import numpy as np
import segmentation as seg
import feature as feat
import clustering as clust

def load_input_image(image_index, folder="train2", path="../data/data_project"):
    filename = "train_{}.png".format(str(image_index).zfill(2))
    
    im = Image.open(os.path.join(path, folder, filename)).convert('RGB')
    im = np.array(im)
    return im

def save_solution_puzzles(image_index, solved_puzzles, outliers, folder="train2", path="../data/data_project", group_id=0):
    path_solution = os.path.join(path, folder + "_solution_{}".format(str(group_id).zfill(2)))
    if not os.path.isdir(path_solution):
        os.mkdir(path_solution)

    print(path_solution)
    for i, puzzle in enumerate(solved_puzzles):
        filename = os.path.join(path_solution, "solution_{}_{}.png".format(str(image_index).zfill(2), str(i).zfill(2)))
        Image.fromarray(puzzle).save(filename)

    for i, outlier in enumerate(outliers):
        filename = os.path.join(path_solution, "outlier_{}_{}.png".format(str(image_index).zfill(2), str(i).zfill(2)))
        Image.fromarray(outlier).save(filename)
        
def solve_and_export_puzzles_image(image_index, folder="train2", path="../data/data_project", group_id="00"):
    """
    Wrapper funciton to load image and save solution
            
    Parameters
    ----------
    image:
        index number of the dataset

    Returns
    """

    # open the image
    image_loaded = load_input_image(image_index, folder=folder, path=path)
    #print(image_loaded)

    ## call functions to solve image_loaded
    solved_puzzles = [(np.random.rand(512, 512, 3) * 255).astype(np.uint8) for i in range(2)]
    outlier_images = [(np.random.rand(128, 128, 3) * 255).astype(np.uint8) for i in range(3)]

    save_solution_puzzles(image_index, solved_puzzles, outlier_images, folder=folder, group_id=group_id)

    return image_loaded, solved_puzzles, outlier_images

def cluster_pieces_from_image(image_path):
    pieces = seg.extract_pieces_from_image(image_path)
    features = feat.extract_features_from_pieces(pieces)
    labels = clust.get_labels_pieces(features)
    
    clust.plot_clustered_pieces(pieces, labels)
    
    return pieces, labels

    