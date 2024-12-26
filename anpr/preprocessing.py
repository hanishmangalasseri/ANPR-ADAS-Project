import cv2

def preprocess_image(image_path):
    """ This function will cover Preprocessing logic, such as grayscale conversion, noise reduction,
    and edge detection"""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)
    return edges
