import cv2
import pytesseract

def recognize_text(plate_image):
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imshow("Edges", gray)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    custom_config = r'--oem 3 --psm 11'
    return pytesseract.image_to_string(thresh, config=custom_config)