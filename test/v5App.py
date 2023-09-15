import preprocess as pp
import display
import load
import detection
import cv2
import numpy as np
import matplotlib.pyplot as plt

def main(image, template):
    try:
        image, template = load.image(image, template)
        
        bg_remove_image = pp.green_screen_removal(image)
        display.show_image(bg_remove_image, "RESULT")
        
        canny_image = detection.edge_detection(bg_remove_image, 0, 255)
        display.show_image(canny_image, "canny")
        
        contour_image, page = detection.contour_detection(bg_remove_image, canny_image)
        display.show_image(contour_image, "contour")
        
        corners, corners_image = detection.find_corner(contour_image, page)
        display.show_image(corners_image, "corner")
        
        cropped_image = pp.crop_image(image, corners[0], corners[2])
        display.show_image(cropped_image, "cropped")

    except Exception as e:
        print("An error occurred:", str(e))
    finally:
        display.exit_app()

if __name__ == "__main__":
    example = input("Enter number of Example: ")
    IMAGE_PATH = '../image/' + example + '.jpg'
    # IMAGE_PATH = '../image/1.jpg'
    print("Load image: " + IMAGE_PATH)
    TEMPLATE_PATH = '../template/idCard.png'
    main(IMAGE_PATH, TEMPLATE_PATH)
