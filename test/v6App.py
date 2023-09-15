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

def setup(image_path, template_path):
    image, template = load.load_image(image_path, template_path)
    image_height, image_width, _ = image.shape
    template_height, template_width, _ = template.shape
    
    print("Image size: {} x {} pixels".format(image_width, image_height))
    print("Template size: {} x {} pixels".format(template_width, template_height))
    
    if template_height > image_height or template_width > image_width:
        print("Template size is bigger than the image. Exiting the app.")
        sys.exit(1)

    return image, template

def pre_process(image, template):
    print("-------- PRE-PROCESS ---------")
    img = image.copy()
    tem = template.copy()

    img_result, tem_result = img, tem
    return img_result, tem_result

def main(image, template):
    try:
        image, template = setup(image, template)
        image_pp, template_pp = pre_process(image, template)
        
        display.show_image(image_pp, "pre-process")
        display.show_image(template_pp, "template pre-process")
    
    except Exception as e:
        print("An error occurred:", str(e))
    finally:
        display.exit_app()

if __name__ == "__main__":
    print("----------- SETUP ------------")
    example = input("Select Image: ")
    IMAGE_PATH = '../image/' + example + '.jpg'
    print("Selected: ", IMAGE_PATH)
    # IMAGE_PATH = '../image/1.jpg'
    TEMPLATE_PATH = '../template/idCard.png'
    main(IMAGE_PATH, TEMPLATE_PATH)
