import cv2
import random
import string

def show_image(image, window_name):
    return cv2.imshow(window_name, image)

def random_window_name(length=10):
    return ''.join(random.choice(string.ascii_letters) for _ in range(length))

def show_image_random(image):
    window_name = random_window_name()
    cv2.imshow(window_name, image)

def exit_app():
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("\n---------- EXIT APP ----------")