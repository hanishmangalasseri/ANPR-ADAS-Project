from preprocessing import preprocess_image
from detection import detect_plate
from recognition import recognize_text
import cv2
class ANPR:
    def __init__(self, image_path):
        self.image_path = image_path
        self.original_image = cv2.imread(image_path)
        self.edges = None
        self.plate_image = None

    def preprocess(self):
        self.edges = preprocess_image(self.image_path)

    def detect(self):
        self.plate_image = detect_plate(self.edges, self.original_image)

    def recognize(self):
        self.detected_text = recognize_text(self.plate_image)

    def run(self):
        self.preprocess()
        self.detect()
        self.recognize()
        print("Detected License Plate Text:", self.detected_text)

class_obj = ANPR("D:\ANPR-ADAS-Project\examples\img1.png")

class_obj.run()