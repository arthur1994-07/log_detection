import cv2
from matplotlib import pyplot as plt


def load_image(img_url):
    if img_url is None:
        print(f"invalid image url")
        return
    return cv2.imread(img_url)


def resize_window(img):
    return cv2.resize(img, (960, 540))


def read_and_plot_img(img_url):
    if img_url is None:
        print(f"invalid image url")
        return
    img = cv2.imread(img_url)
    if img is None:
        print(f"Error: Could not open or read image at {img_url}")
        return
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    edges = cv2.Canny(img_gray, 30, 100)

    plt.subplot(1, 1, 1)
    plt.imshow(edges)
    plt.show()
