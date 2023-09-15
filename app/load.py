import cv2

def load_image(image_path, template_path):
    image = cv2.imread(image_path)
    template = cv2.imread(template_path)
    return image, template

