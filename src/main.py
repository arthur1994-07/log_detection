from src.algorithm import polygon_detection as corner_detect

num_logs = corner_detect.count_logs("assets/log_2.jpg")
print(num_logs)