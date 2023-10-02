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

def Setup(image_path, template_path):
    image, template = load.load_image(image_path, template_path)
    return image, template

def PreProcess(image, template):
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
    return process_image, process_template


def main(image, template):
    try:
        start_time = time.time() 
        image, template = Setup(image, template)
        pp_image, pp_template = PreProcess(image, template)

        print("\n------------ OCR ------------\n")
        print("  - Display Image.")
        display.show_image(pp_image, "image")
        display.show_image(pp_template, "template")

    except Exception as e:
        print("  !! ERROR:", str(e))
    finally:
        elapsed_time = time.time() - start_time 
        print("\nElapsed Time: {:.2f} seconds".format(elapsed_time))
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