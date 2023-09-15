import cv2
import detection
import numpy as np

def resize_image(image, x: float, y: float):
    return cv2.resize(image, (0, 0), fx = x, fy = y)

def gray_image(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def morph_image(image, iterations: int):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations = iterations)

def median_blur_image(image):
    return cv2.medianBlur(image, 5)

def invert_image(image):
    return cv2.bitwise_not(image)

def thresh_image(image, x: int, y: int):
    thresholded = cv2.threshold(image, x, y, cv2.THRESH_BINARY)[1]
    return thresholded

def adaptive_thresh_image(image, maxValue: int, blockSize: int, C: int):
    thresholded = cv2.adaptiveThreshold(image, maxValue, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize, C)
    return thresholded

def enhance_contrast_bw(image):
    return cv2.equalizeHist(image)

def enhance_contrast_color(image):
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8,8))

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)  # split on 3 different channels

    l2 = clahe.apply(l)  # apply CLAHE to the L-channel

    lab = cv2.merge((l2,a,b))  # merge channels
    res = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
    return res

def grab_cut_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (20, 20, image.shape[1] - 20, image.shape[0] - 20)
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    return image * mask2[:, :, np.newaxis]

def draw_id_card_rectangle(image, top_left, bottom_right):
    img = image.copy()
    color = (255, 255, 125)
    thick = 2
    img = cv2.rectangle(image, top_left, bottom_right, color, thick)
    return img

def crop_image(image, top_left, bottom_right):
    top_left = (max(0, top_left[0]), max(0, top_left[1]))
    bottom_right = (min(image.shape[1], bottom_right[0]), min(image.shape[0], bottom_right[1]))

    cropped_image = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    return cropped_image

def remove_shadow(image):
    rgb_planes = cv2.split(image)
        
    result_planes = []
    result_norm_planes = []
    
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)
        
    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)
    
    return result, result_norm

def green_screen_removal(image):
    lower = np.array([35, 50, 50])  # Lower bound for green in HSV color space
    upper = np.array([85, 255, 255])  # Upper bound for green in HSV color space
    
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    thresh = cv2.inRange(image_hsv, lower, upper)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    mask = 255 - morph
    
    result = cv2.bitwise_and(image, image, mask=mask)
    return result