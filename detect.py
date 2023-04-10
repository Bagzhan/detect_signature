import cv2
import numpy as np
from skimage import measure, morphology
from skimage.measure import regionprops
from crop_image import ImageCropper

def make_mask(image):
    """
    create a mask that the bright parts are marked as 255, the rest as 0.

    params
    ------
    image: numpy array

    return
    ------
    frame_threshold: numpy array
    """
    low_threshold=(0, 0, 250)
    high_threshold=(255, 255, 255)
    frame_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    frame_threshold = cv2.inRange(
        frame_HSV, low_threshold, high_threshold
    )
    return frame_threshold


def extract(mask):
    """
    params
    ------
    mask: numpy array

    return
    ------
    labeled_image: numpy array
        The labeled image.
    """
    condition = mask > mask.mean()
    labels = measure.label(condition, background=1)

    total_pixels = 0
    nb_region = 0
    average = 0.0
    for region in regionprops(labels):
        if region.area > 10:
            total_pixels += region.area
            nb_region += 1
    
    if nb_region > 1:
        average = total_pixels / nb_region
        # small_size_outlier is used as a threshold value to remove pixels
        # are smaller than small_size_outlier
        small_size_outlier = average * 3 + 100

        # big_size_outlier is used as a threshold value to remove pixels
        # are bigger than big_size_outlier
        big_size_outlier = small_size_outlier * 15

        # remove small pixels
        labeled_image = morphology.remove_small_objects(labels, small_size_outlier)
        # remove the big pixels
        component_sizes = np.bincount(labeled_image.ravel())
        too_small = component_sizes > (big_size_outlier)
        too_small_mask = too_small[labeled_image]
        labeled_image[too_small_mask] = 0

        labeled_mask = np.full(labeled_image.shape, 255, dtype="uint8")
        labeled_mask = labeled_mask * (labeled_image == 0)
    else:
        labeled_mask = mask

    return labeled_mask


def detect_signature(image):
    # Create a binary mask of the input image using adaptive thresholding.
    mask = make_mask(image)
    #Extract connected components from a binary mask and label them.
    labeled_mask = extract(mask)
    #Initialize the ImageCropper with the minimum region size and border ratio.
    cropper = ImageCropper()
    
    results = cropper.run(labeled_mask)
    return results