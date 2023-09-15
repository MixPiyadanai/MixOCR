import cv2
import preprocess as pp
import display
import numpy as np

def edge_detection(image, t_lower, t_upper):
    edges = cv2.Canny(image, t_lower, t_upper)
    return edges

def contour_detection(image, edges):
    con = np.zeros_like(image)
    
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    page = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    con = cv2.drawContours(con, page, -1, (0, 255, 255), 3)
    return con, page

def find_corner(image, page):
    con = np.zeros_like(image)

    for c in page:

        epsilon = 0.02 * cv2.arcLength(c, True)
        corners = cv2.approxPolyDP(c, epsilon, True)

        if len(corners) == 4:
            break
        
    cv2.drawContours(con, c, -1, (0, 255, 255), 3)
    cv2.drawContours(con, corners, -1, (0, 255, 0), 10)

    corners = sorted(np.concatenate(corners).tolist())
    corners = order_points(corners)

    for index, c in enumerate(corners):
        character = chr(65 + index)
        cv2.putText(con, character, tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
    return corners, con

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
    # Return the ordered coordinates.
    return rect.astype('int').tolist()

def match_template_tlbr(image, template):
    img_copy = image.copy()
    img_copy = pp.gray_image(img_copy)
    template = pp.gray_image(template)
    
    w, h = template.shape[::-1]
    method = cv2.TM_CCOEFF_NORMED
    
    res = cv2.matchTemplate(img_copy, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    
    print("tl: ", top_left, "br: ", bottom_right)
    
    return image, top_left, bottom_right
