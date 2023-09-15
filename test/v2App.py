import cv2
from imutils.perspective import four_point_transform
import numpy as np

IMAGE_SOURCE = 'image/mix.jpg'

def load_image(src):
    return cv2.imread(src)

def resize_image(image):
    return cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

def noise_removal(image):
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return (image)

def process_image(image):
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('GRAY', gray)
    
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    hsv_image[..., 1] = cv2.add(hsv_image[..., 1], 100) 
    
    saturated_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    cv2.imshow('SATURATED', saturated_image)
    
    # inverted = cv2.bitwise_not(saturated_image)
    # cv2.imshow('INVERTED', inverted)
    
    no_noise = noise_removal(saturated_image)
    cv2.imshow('NONOISE', no_noise)
    
    blur = cv2.GaussianBlur(no_noise, (5, 5), 0)
    cv2.imshow('BLURRED', blur)
    edged = cv2.Canny(blur, 75, 200)
    return edged


def find_contours(image):
    contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    doc_cnts = None
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.05 * peri, True)
        
        if len(approx) == 4:
            doc_cnts = approx
            break
    
    return doc_cnts

def draw_contours_on_image(image, contour):
    return cv2.drawContours(image, [contour], -1, (0, 255, 255), 1)

def apply_perspective_transform(image, contour):
    warped = four_point_transform(image, contour.reshape(4, 2))
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    return warped

def exit_app():
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("EXIT")

def main():
    try:
        image = load_image(IMAGE_SOURCE)
        image = resize_image(image)
        cv2.imshow('SOURCE', image)
        
        processed_image = process_image(image)
        cv2.imshow('PROCESSED', processed_image)
        
        contours = find_contours(processed_image)
        contour_image = draw_contours_on_image(image.copy(), contours)
        cv2.imshow("CONTOURS", contour_image)
        
        scanned_image = apply_perspective_transform(image, contours)
        cv2.imshow("WARPPED", scanned_image)
            
    except Exception as e:
        print("An error occurred:", str(e))
    finally:
        exit_app()

if __name__ == "__main__":
    main()
