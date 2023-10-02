import pytesseract
import cv2
import re
import json

def startDraw(image):
    img = image.copy()

    dyPosition = 0
    attempts = 0
    isIdFound = False
    
    while attempts < 3:
        idNum_img, idNum, isIdFound = idNumber(img, 50 + dyPosition, isIdFound)
        if len(idNum) == 13:
            name_img, name = Name(img, 110 + dyPosition)
            # Filter out non-Thai characters using regex
            name = re.sub(r'[^\u0E00-\u0E7F\s]+', '', name)
            result = {
                "idNum": idNum,
                "name": name.strip()
            }
            result = json.dumps(result, indent=4, ensure_ascii=False)
            print("\n", result)
            return idNum_img, idNum
        else:
            dyPosition += 20
            attempts += 1
            print("  - Attempts: " , attempts)

    print("Bad Image.")
    return idNum_img, None

def idNumber(image, y, isIdFound):
    rect_color = (0, 255, 0) 
    rect_thickness = 2
    rect_x1 = 510
    rect_y1 = y
    rect_x2 = 1000
    rect_y2 = y + 60

    roi = image[rect_y1:rect_y2, rect_x1:rect_x2]

    text = pytesseract.image_to_string(roi)

    numeric_text = re.sub(r'\D', '', text)

    if len(numeric_text) == 13:
        isIdFound = True
        
    if isIdFound:
        cv2.rectangle(image, (rect_x1, rect_y1), (rect_x2, rect_y2), rect_color, rect_thickness)
    
    return image, numeric_text, isIdFound

def Name(image, y):
    rect_color = (0, 255, 0) 
    rect_thickness = 2
    rect_x1 = 315
    rect_y1 = y
    rect_x2 = 1100
    rect_y2 = y + 95

    roi = image[rect_y1:rect_y2, rect_x1:rect_x2]

    text = pytesseract.image_to_string(roi, lang="tha")
    
    cv2.rectangle(image, (rect_x1, rect_y1), (rect_x2, rect_y2), rect_color, rect_thickness)

    return image, text