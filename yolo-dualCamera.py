import cv2
import threading
from ultralytics import YOLO

model = YOLO('oppo-ai.pt')

def detect_camera(cam_id, name):
    cap = cv2.VideoCapture(cam_id)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, show=False)  
        annotated_frame = results[0].plot() 
        cv2.imshow(name, annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

thread1 = threading.Thread(target=detect_camera, args=(0, 'Kamera 1'))
thread2 = threading.Thread(target=detect_camera, args=(1, 'Kamera 2'))

thread1.start()
thread2.start()

thread1.join()
thread2.join()

cv2.destroyAllWindows()
