import cv2
from ..util.image_helper import load_image, resize_window
import numpy as np

def count_logs(image_path):
    try:
        # 1. Load the image
        img = load_image(image_path)
        if img is None:
            print(f"Error: Could not open or read image at {image_path}")
            return 0
        # 2. Preprocess the image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Adjust kernel size (5,5) as needed
        edged = cv2.Canny(blurred, 30, 150)  # Adjust thresholds (30, 150) as needed

        # 3. Find contours
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 4. Filter contours (remove noise, focus on log shapes)
        min_contour_area = 1000# Adjust this value based on the typical size of logs in your images
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

        # 5. Count the logs
        num_logs = len(filtered_contours)

        # 6. Optional: Draw contours on the original image to visualize the result
        cv2.drawContours(img, filtered_contours, -1, (0, 255, 0), 2)  # Green contours

        cv2.imshow("Contours", resize_window(img))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return num_logs
    except Exception as e:
        print(f"An error occurred: {e}")
    return 0



def rectangle_detect(img_url):

    img = cv2.imread(img_url)



