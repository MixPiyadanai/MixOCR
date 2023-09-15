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
    
    gray_image = pp.gray_image(process_image)
    gray_template = cv2.cvtColor(process_template, cv2.COLOR_BGR2GRAY)
    print("\n  + Gray Scale.")
    
    median_blur_image = pp.median_blur_image(gray_image)
    median_blur_template = pp.median_blur_image(gray_template)
    print("  + Median Blur.")
    
    adaptive_threshold_image = cv2.adaptiveThreshold(median_blur_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 85, 11)
    adaptive_threshold_template = cv2.adaptiveThreshold(median_blur_template, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 85, 11)
    print("  + Adaptive Threshold.")
    
    # kernel = np.ones((1, 2), np.uint8)
    # dilation_image = cv2.dilate(adaptive_threshold_image, kernel, iterations=1)
    # dilation_template = cv2.dilate(adaptive_threshold_template, kernel, iterations=1)
    # print("  + Dilation.")
    
    return adaptive_threshold_image, adaptive_threshold_template


def main(image, template):
    try:
        image, template = Setup(image, template)
        pp_image, pp_template = PreProcess(image, template)
        
        print("\n----------- MAIN ------------\n")
        print("  - Display Image.")
        display.show_image(pp_image, "image")
        display.show_image(pp_template, "template")
    
    except Exception as e:
        print("An error occurred:", str(e))
    finally:
        display.exit_app()

if __name__ == "__main__":
    print("\n----------- SETUP ------------\n")
    example = input("Select Image: ")
    IMAGE_PATH = '../image/' + example + '.jpg'
    print("Selected: ", IMAGE_PATH)
    # IMAGE_PATH = '../image/1.jpg'
    TEMPLATE_PATH = '../template/idCard.png'
    main(IMAGE_PATH, TEMPLATE_PATH)
