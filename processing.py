import cv2
import numpy as np
import imutils
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

# Constant Default Values
MIN_BLUR_SIZE = 30
KERNEL_SIZE = 11
DEAD_KERNEL_SIZE = 3
MIN_AREA = 100
EPSILON = 200
DEAD_CELL_THRESH = 0.65


def drawContours(img, contours, color):
    for contour in contours:
        ellipse = cv2.fitEllipse(contour)
        cv2.ellipse(img, ellipse, color=color, thickness=2)
    return img


def find_contours(img, minArea):
    found_contours = []
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
        for contour in contours:
            if len(contour) >= 5 and cv2.contourArea(contour) > minArea:
                found_contours.append(contour)
    return found_contours


def morphological_operation(img, kernel_size):
    """Do morphological operations on the image"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mod_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    mod_img = cv2.morphologyEx(mod_img, cv2.MORPH_CLOSE, kernel)
    return mod_img


def image_blur(img, size, epsilon):
    guided_filter = cv2.ximgproc.createGuidedFilter(img, size, epsilon)
    return guided_filter.filter(img)


def flood_fill(img):
    # Copy the thresholded image.
    floodfill = img.copy()
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    height, width = img.shape[:2]
    mask = np.zeros((height + 2, width + 2), np.uint8)
    # Floodfill from point (0, 0)
    cv2.floodFill(floodfill, mask, (0, 0), 255)
    # Invert floodfilled image
    floodfill = cv2.bitwise_not(floodfill)
    return floodfill


def const_ench(img):
    """Adjust brightness of an image"""
    clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(8, 8))
    final = clahe.apply(img)
    return final


def deadcells_detection(img, black_area_mask, kernel_size=DEAD_KERNEL_SIZE, deadcell_thresh=DEAD_CELL_THRESH):
    """Detect dead cells in the given image"""
    const_img = const_ench(img)
    final_gray = cv2.bitwise_and(const_img, black_area_mask)
    otsu = cv2.threshold(final_gray[final_gray != 0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    mask = cv2.threshold(final_gray, otsu*deadcell_thresh, 255, cv2.THRESH_BINARY_INV)[1]
    mask = cv2.bitwise_and(mask, black_area_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    modMask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    return modMask


def count_cells(alive_cell_mask, dead_cell_mask, img, minArea=MIN_AREA):
    drawImg = img.copy()
    alive_cells = []
    dead_cells = []
    contours, hierarchy = cv2.findContours(alive_cell_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) != 0:
        for contour in contours:
            if len(contour) >= 5 and cv2.contourArea(contour) > minArea:
                ellipse = cv2.fitEllipse(contour)
                cv2.ellipse(drawImg, ellipse, color=(0, 0, 255), thickness=2)
                alive_cells.append(contour)

    contours, hierarchy = cv2.findContours(dead_cell_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
        for contour in contours:
            if len(contour) >= 5 and cv2.contourArea(contour) > minArea:
                ellipse = cv2.fitEllipse(contour)
                cv2.ellipse(drawImg, ellipse, color=(0, 255, 0), thickness=2)
                dead_cells.append(contour)

    return drawImg, len(alive_cells), len(dead_cells)


def watershed_count(image, drawImg, dead_cell_mask, minArea):
    alive_cells = []
    distance = ndi.distance_transform_edt(image)
    coords = peak_local_max(distance, min_distance=8, labels=image)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=image)
    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(image.shape, dtype="uint8")
        mask[labels == label] = 255
        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        if len(c) >= 5 and cv2.contourArea(c) > minArea:
            alive_cells.append(c)
    dead_cells = find_contours(dead_cell_mask, minArea)
    drawContours(drawImg, alive_cells, (0, 0, 255))
    drawContours(drawImg, dead_cells, (0, 255, 0))
    return drawImg, len(alive_cells), len(dead_cells)


def process(img, *, kernelSize= KERNEL_SIZE, _minArea= MIN_AREA, _blursize=MIN_BLUR_SIZE,
            _epsilon=EPSILON, _deadcellThresh=DEAD_CELL_THRESH, _deadKernelSize=DEAD_KERNEL_SIZE):
    """Find dead and alive cells in the given image using the given parameters"""
    kernelSize = KERNEL_SIZE if kernelSize == '' else int(kernelSize)   # If no value is given equilize the value to default value
    _minArea = MIN_AREA if _minArea == '' else float(_minArea)
    _blursize= MIN_BLUR_SIZE if _blursize == '' else int(_blursize)
    _epsilon = EPSILON if _epsilon == '' else int(_epsilon)
    _deadcellThresh = DEAD_CELL_THRESH if _deadcellThresh == '' else int(_deadcellThresh)
    _deadKernelSize = DEAD_KERNEL_SIZE if _deadKernelSize == '' else int(_deadKernelSize)
    if _blursize < 1:
        raise Exception("Blur Size cannot be smaller than 1")
    if kernelSize < 1:
        raise Exception("Kernel size cannot be smaller than 1")
    if _deadcellThresh < 0 or _deadcellThresh > 255:
        raise Exception("Dead cell threshold value is not within the boundaries")
    if _deadKernelSize < 1:
        raise Exception("Dead cell detection kernel size cannot be smaller than 1")
    #print(f'blursize is {_blursize}, kernelsize is {kernelSize}, minArea is {_minArea}, \
    #    epsilon is {_epsilon}, deadcellThresh is {_deadcellThresh}, \
    #    deadcellKernelSize is {_deadKernelSize}')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    black_area = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)[1]
    black_area_morph = morphological_operation(black_area, 30)

    denoise_img = cv2.fastNlMeansDenoising(gray)
    dead_cell_result = deadcells_detection(denoise_img, black_area_morph, kernel_size=_deadKernelSize, deadcell_thresh=_deadcellThresh)  # Detect dead cells
    blur_img = image_blur(denoise_img, _blursize, _epsilon)
    ench_img = const_ench(blur_img)  # Brightness Adjustment
    ench_img = cv2.bitwise_and(ench_img, black_area_morph)
    otsu = cv2.threshold(ench_img[ench_img != 0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    mask = cv2.threshold(ench_img, otsu, 255, cv2.THRESH_BINARY)[1]
    mask = cv2.bitwise_or(mask, flood_fill(mask))
    morphed_mask = morphological_operation(mask, kernelSize)

    # resultImg, len_alive_cell, len_dead_cell = count_cells(morphed_mask, dead_cell_result, img)
    resultImg, len_alive_cell, len_dead_cell = watershed_count(morphed_mask, img.copy(), dead_cell_result, _minArea)
    return resultImg, len_alive_cell, len_dead_cell