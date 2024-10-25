import cv2, time 
video = cv2.VideoCapture(0)
check, frame = video.read()
print(check)
video.release()