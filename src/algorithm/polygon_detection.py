import cv2
from ..util.ImageHelper import ImageHelper
import numpy as np


def count_logs(image_path):
    try:
        # 1. Load the image
        img = ImageHelper.load_image(image_path)
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
        min_contour_area = 150  # Adjust this value based on the typical size of logs in your images
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

        # 5. Count the logs
        num_logs = len(filtered_contours)

        # 6. Optional: Draw contours on the original image to visualize the result
        cv2.drawContours(img, filtered_contours, -1, (0, 255, 0), 2)  # Green contours

        cv2.imshow("Contours", ImageHelper.resize_window(img))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return num_logs
    except Exception as e:
        print(f"An error occurred: {e}")
    return 0


def rectangle_detect(img_url):
    img = ImageHelper.load_image(img_url)
    if img is None:
        print(f"Error: Could not open or read image at {img_url}")
        return 0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpen = cv2.filter2D(blur, -1, sharpen_kernel)

    thresh = cv2.threshold(sharpen, 183, 255, cv2.THRESH_BINARY)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    close = cv2.morphologyEx(thresh, cv2.MORPH_RECT, kernel, iterations=3)

    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    min_area = 50
    max_area = 1500
    image_number = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(c)
            #ROI = img[y:y+h, x:x+w]
            #cv2.imwrite('ROI_{}.png'.format(image_number), ROI)
            cv2.rectangle(img, (x, y), (x + w, y + h), (36, 255, 12), 2)
            image_number += 1

    print(image_number)
    #cv2.imshow('sharpen', ImageHelper.resize_window(sharpen))
    cv2.imshow('close', ImageHelper.resize_window(close))
    cv2.imshow('thresh', ImageHelper.resize_window(thresh))
    cv2.imshow('image', ImageHelper.resize_window(img))
    cv2.waitKey()


def detect(img_url):
    if img_url is None: return
    img = ImageHelper.load_image(img_url)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    blurred = cv2.GaussianBlur(blur, (5, 5), 0)
    
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpen = cv2.filter2D(blurred, -1, sharpen_kernel)

    thresh = cv2.threshold(sharpen, 183, 255, cv2.THRESH_BINARY)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    close = cv2.morphologyEx(thresh, cv2.MORPH_RECT, kernel, iterations=1)

    contours, _ = cv2.findContours(close.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_contour_area = 200  # Adjust this value based on the typical size of logs in your images
    #filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    filtered_contours = []
    for cnt in contours:
        ep = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, ep, True)
        print(len(approx))

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 5)

        area = cv2.contourArea(cnt)
        if area > min_contour_area:
            filtered_contours.append(cnt)

    num_logs = len(filtered_contours)

    #cv2.drawContours(img, filtered_contours, -1, (0, 255, 0), 2)  # Green contours

    cv2.imshow("Contours", ImageHelper.resize_window(img))
    cv2.imshow("close", ImageHelper.resize_window(close))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return num_logs


# AI generated code
def detect_logs(image_path):
    if image_path is None: return
    image = ImageHelper.load_image(image_path)

    # Convert the image to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours from the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw rectangles around detected contours
    for contour in contours:
        # Approximate contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        # Check if the contour is rectangular
        if len(approx) == 4:  # Rectangles have 4 vertices
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Optionally display the image window
    cv2.imshow("Detected Logs", ImageHelper.resize_window(image))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

