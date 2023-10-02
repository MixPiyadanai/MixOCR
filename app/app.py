import preprocess as pp
import display
import load
import detection
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import pytesseract
import sys
import os
import time

def scan(img):
    # Resize image to workable size
    dim_limit = 1080
    max_dim = max(img.shape)
    if max_dim > dim_limit:
        resize_scale = dim_limit / max_dim
        img = cv2.resize(img, None, fx=resize_scale, fy=resize_scale)
    # Create a copy of resized original image for later use
    orig_img = img.copy()
    # Repeated Closing operation to remove text from the document.
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=3)
    # GrabCut
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (20, 20, img.shape[1] - 20, img.shape[0] - 20)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]
 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (11, 11), 0)
    # Edge Detection.
    canny = cv2.Canny(gray, 0, 200)
    canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
 
    # Finding contours for the detected edges.
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # Keeping only the largest detected contour.
    page = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
 
    # Detecting Edges through Contour approximation.
    # Loop over the contours.
    if len(page) == 0:
        return orig_img
    for c in page:
        # Approximate the contour.
        epsilon = 0.02 * cv2.arcLength(c, True)
        corners = cv2.approxPolyDP(c, epsilon, True)
        # If our approximated contour has four points.
        if len(corners) == 4:
            break
    # Sorting the corners and converting them to desired shape.
    corners = sorted(np.concatenate(corners).tolist())
    # For 4 corner points being detected.
    corners = order_points(corners)
 
    destination_corners = find_dest(corners)
 
    h, w = orig_img.shape[:2]
    # Getting the homography.
    M = cv2.getPerspectiveTransform(np.float32(corners), np.float32(destination_corners))
    # Perspective transform using homography.
    final = cv2.warpPerspective(orig_img, M, (destination_corners[2][0], destination_corners[2][1]),
                                flags=cv2.INTER_LINEAR)
    return final

def order_points(pts):
    '''Rearrange coordinates to order:
      top-left, top-right, bottom-right, bottom-left'''
    rect = np.zeros((4, 2), dtype='float32')
    pts = np.array(pts)
    s = pts.sum(axis=1)
    # Top-left point will have the smallest sum.
    rect[0] = pts[np.argmin(s)]
    # Bottom-right point will have the largest sum.
    rect[2] = pts[np.argmax(s)]
 
    diff = np.diff(pts, axis=1)
    # Top-right point will have the smallest difference.
    rect[1] = pts[np.argmin(diff)]
    # Bottom-left will have the largest difference.
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect.astype('int').tolist()

def find_dest(pts):
    (tl, tr, br, bl) = pts
    # Finding the maximum width.
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
 
    # Finding the maximum height.
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # Final destination co-ordinates.
    destination_corners = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]
 
    return order_points(destination_corners)

def resize_image(image, template):
    print("\n-------- PRE-PROCESS ---------")
    process_image = image.copy()
    process_template = template.copy()
    
    image_height, image_width, _ = process_image.shape
    template_height, template_width, _ = process_template.shape
    
    print("\n  - Image size: {} x {} pixels.".format(image_width, image_height))
    print("  - Template size: {} x {} pixels.".format(template_width, template_height))
    
    if template_height > image_height or template_width > image_width:
        
        print("\n  ! Template size is bigger than the image. Resizing image...")
        
        aspect_ratio = template_width / float(image_width)
        new_height = int(image_height * aspect_ratio)
        process_image = cv2.resize(process_image, (template_width, new_height))
        
        print("  - Resized Image size: {} x {} pixels.".format(template_width, new_height))
    return process_image

def draw_rectangle(image):
    rect_color = (0, 255, 0) 
    rect_thickness = 2
    rect_x1 = 510
    rect_y1 = 75
    rect_x2 = 1000
    rect_y2 = 145

    cv2.rectangle(image, (rect_x1, rect_y1), (rect_x2, rect_y2), rect_color, rect_thickness)

    roi = image[rect_y1:rect_y2, rect_x1:rect_x2]

    text = pytesseract.image_to_string(roi)

    return image, text


def Setup(image_path, template_path):
    image, template = load.load_image(image_path, template_path)
    return image, template

def main(image, template):
    try:
        image, template = Setup(image, template)

        print("\n------------ OCR ------------\n")
        print("  - Display Image.")
        crop_img = scan(image)
        resize_img = resize_image(crop_img, template)
        image_with_rectangle, text = draw_rectangle(resize_img)
        display.show_image(image_with_rectangle, "image")
        display.show_image(template, "template")
        print("  - " + text)

    except Exception as e:
        print("  !! ERROR:", str(e))
    finally:
        display.exit_app()
        
if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    print("\n----------- SETUP ------------\n")
    example = input("  - Select Image: ")
    IMAGE_PATH = '../image/' + example + '.jpg'
    print("  - Selected: ", IMAGE_PATH)
    # IMAGE_PATH = '../image/1.jpg'
    TEMPLATE_PATH = '../template/idCard.png'
    main(IMAGE_PATH, TEMPLATE_PATH)