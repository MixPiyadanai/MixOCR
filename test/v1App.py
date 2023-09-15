import cv2
import pytesseract
import numpy as np
import random

TEMPLATE_ID_CARD = 'template/idCard.png'
TEMPLATE_HARD_POINT = 'template/hard_point.png'
IMAGE_PATH = 'image/main.jpg'
HARD_POINT_POSITION = [0, 0]

def read_image(src):
    return cv2.imread(src)

def resize_image(image, x, y):
    return cv2.resize(image, (0, 0), fx=x, fy=y)

def display_image(image, window_name):
    cv2.imshow(window_name, image)
    
def cropped_image_template(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    template = cv2.imread('template/idCard.png', cv2.IMREAD_GRAYSCALE)
    
    w, h = template.shape[::-1]
    
    if w > image_gray.shape[1] or h > image_gray.shape[0]:
        print("Template dimensions are larger than image dimensions.")
        return None
    
    res = cv2.matchTemplate(image_gray, template, cv2.TM_CCOEFF_NORMED)
    
    _, _, _, maxLoc = cv2.minMaxLoc(res)
    crop_img = image[maxLoc[1]:maxLoc[1] + h, maxLoc[0]:maxLoc[0] + w, :] 

    return crop_img

def grayscale_image(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def inverted_image(image):
    return cv2.bitwise_not(image)

def thresh_image(image, black, white):
    # maximum value = 255
    _, thresholded = cv2.threshold(image, black, white, cv2.THRESH_BINARY)
    return thresholded

def set_hard_point_postion(x, y):
    HARD_POINT_POSITION[0] = x
    HARD_POINT_POSITION[1] = y
    print("Hard Point:", HARD_POINT_POSITION)

def get_id_number(image):
    thresh = thresh_image(image, 130, 255)
    
    scan_range = 40
    
    for i in range(50):
        startX = 485 + random.randint(-scan_range, scan_range)
        startY = 80 + random.randint(-scan_range, scan_range)
        endX = startX + 355
        endY = startY + 45
        
        print("attempt ", i)
        print("StartX:", startX, "EndX:", endX )
        print("Y:", startY, "EndY:", endY)
        
        cropped_id_area = thresh[startY:endY, startX:endX]
        rgb_cropped = cv2.cvtColor(cropped_id_area, cv2.COLOR_GRAY2RGB)
            
        id_text = pytesseract.image_to_string(rgb_cropped)
        numeric_id = ''.join(filter(str.isdigit, id_text))
        display_image(rgb_cropped, 'cropped_id_area')
        
        print(numeric_id)
        if numeric_id.isdigit() and len(numeric_id) == 13:
            print("found (exit loop)")
            cv2.rectangle(image, (startX, startY), (endX, endY), (255, 255, 255), 1)
            
            set_hard_point_postion(startX - 350, startY) 
            cv2.circle(image, (HARD_POINT_POSITION[0], HARD_POINT_POSITION[1]), 5, (255,0,0), -1)
            
            break
        print("-----")
    
    return image, numeric_id

def get_name(image):
    
    name = "HELELO"
    return image, name

def exit_app():
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("EXIT")

def main():
    try:
        image = read_image(IMAGE_PATH)
        
        cropped_image = cropped_image_template(image)
        grayscale = grayscale_image(cropped_image)
        invert = inverted_image(grayscale)

        id_area, id = get_id_number(invert)
        name_area, name = get_name(invert)

        display_image(id_area, 'id_area')

        if id.isdigit() and len(id) == 13:
            export_data = {
                "identification_number": id,
                "name": name
            }
            print(export_data)
        else:
            print("Invalid data. Exiting...")

    except Exception as e:
        print("An error occurred:", str(e))

    finally:
        exit_app()

if __name__ == "__main__":
    main()