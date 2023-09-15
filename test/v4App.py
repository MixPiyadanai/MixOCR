import cv2
import numpy as np

def exit_app():
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("EXIT")

def load_and_preprocess_images(image_path, template_path):
    image = cv2.imread(image_path, 0)
    template = cv2.imread(template_path, 0) 
    return image, template

def match_template(image, template):
    temp_w, temp_h = template.shape[::-1]
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = max_loc
    bottom_right = (top_left[0] + temp_w, top_left[1] + temp_h)
    return top_left, bottom_right

def draw_rectangle(image, top_left, bottom_right):
    detected = cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 2)
    detected = cv2.resize(detected, (0, 0), fx=0.5, fy=0.5)
    return detected

def crop_detected_region(image, top_left, bottom_right):
    cropped_region = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    return cropped_region

def warp_image(image):
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.morphologyEx(color_image, cv2.MORPH_CLOSE, kernel, iterations=3)

    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (20, 20, img.shape[1] - 20, img.shape[0] - 20)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (11, 11), 0)
    canny = cv2.Canny(gray, 0, 200)
    canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    
    con = np.zeros_like(img)
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    page = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    con = cv2.drawContours(con, page, -1, (0, 255, 255), 3)
    
    con = np.zeros_like(img)

    for c in page:
        epsilon = 0.02 * cv2.arcLength(c, True)
        corners = cv2.approxPolyDP(c, epsilon, True)
        if len(corners) == 4:
            break
    cv2.drawContours(con, c, -1, (0, 255, 255), 3)
    cv2.drawContours(con, corners, -1, (0, 255, 0), 10)
    corners = sorted(np.concatenate(corners).tolist())
    
    corners = order_points(corners)
    
    destination_corners = find_dest(corners)
    
    h, w = image.shape[:2]
    M = cv2.getPerspectiveTransform(np.float32(corners), np.float32(destination_corners))
    final = cv2.warpPerspective(image, M, (destination_corners[2][0], destination_corners[2][1]), flags=cv2.INTER_LINEAR)
    return final

def order_points(pts):
    rect = np.zeros((4, 2), dtype='float32')
    pts = np.array(pts)
    s = pts.sum(axis=1)
    
    rect[0] = pts[np.argmin(s)]

    rect[2] = pts[np.argmax(s)]
 
    diff = np.diff(pts, axis=1)

    rect[1] = pts[np.argmin(diff)]

    rect[3] = pts[np.argmax(diff)]

    return rect.astype('int').tolist()

def find_dest(pts):
    (tl, tr, br, bl) = pts

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    destination_corners = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]

    return order_points(destination_corners)

def main(image, template):
    try:
        image, template = load_and_preprocess_images(image, template)
        top_left, bottom_right = match_template(image, template)
        
        detected = draw_rectangle(image, top_left, bottom_right)
        cv2.imshow('Detected template', detected)

        cropped_region = crop_detected_region(image, top_left, bottom_right)
        cv2.imshow('Cropped Region', cropped_region)
        
        warped_image = warp_image(cropped_region)
        cv2.imshow('Warp Image', warped_image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print("An error occurred:", str(e))
    finally:
        exit_app()

if __name__ == "__main__":
    IMAGE_PATH = 'image/2.jpg'
    TEMPLATE_PATH = 'template/idCard.png'
    main(IMAGE_PATH, TEMPLATE_PATH)