import cv2
from imutils.perspective import four_point_transform
import numpy as np

IMAGE_PATH = 'image/main.jpg'

# Universal Function
def read_image(src):
    return cv2.imread(src)

def resize_image(image, x, y):
    return cv2.resize(image, (0, 0), fx=x, fy=y)

def display(image, window_name):
    cv2.imshow(window_name, image)

def exit_app():
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("EXIT")
    

def morphology_image(image, iterations):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations= iterations)

def edge_image(image):
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    return cv2.Canny(blur, 75, 200)

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
    con = image.copy()
    for index, c in enumerate(contour):
        x, y = tuple(c[0])
        position = f"({x}, {y})"
        cv2.putText(con, position, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return cv2.drawContours(con, [contour], -1, (255, 255, 0), 2)



def apply_perspective_transform(image, contour):
    warped = four_point_transform(image, contour.reshape(4, 2))
    return warped

def main():
    try:
        original_image = read_image(IMAGE_PATH)
        resized_image = resize_image(original_image, 0.45, 0.45)
        display(resized_image, "SOURCE")
        
        morph_image = morphology_image(resized_image, 20)
        display(morph_image, "MORPH")
        
        edged_image = edge_image(morph_image)
        display(edged_image, "EDGED")
        
        contours = find_contours(edged_image)
        
        if contours is not None and len(contours) == 4:
            contour_image = draw_contours_on_image(resized_image.copy(), contours)
            cv2.imshow("CONTOURS", contour_image)
        else:
            print("No valid contours found.")
            
        scanned_image = apply_perspective_transform(resized_image, contours)
        cv2.imshow("WARPPED", scanned_image)
            
    except Exception as e:
        print("An error occurred:", str(e))

    finally:
        exit_app()

if __name__ == "__main__":
    main()