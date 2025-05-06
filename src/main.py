from src.algorithm import polygon_detection as polygon_detect

img_url = "assets/log_4.jpg"
num_logs = polygon_detect.detect(img_url)

print(num_logs)
#polygon_detect.detect_logs(img_url)
