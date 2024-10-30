from ultralytics import YOLO
import cv2 as cv
import dotenv


model = YOLO('yolov8n.pt')

cap = cv.VideoCapture('videos\\CCTV Footage.mp4')

while True:
    _isTrue, frame = cap.read()
    
    if not _isTrue:
        break

    results = model(frame)

    for result in results:
        for obj in result.boxes.data:  
            x1, y1, x2, y2, conf, cls = obj.tolist()  
            class_name = model.names[int(cls)]  

             
            cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

            
            label = f"{class_name}: {conf:.2f}"
            
            
            cv.putText(frame, label, (int(x1), int(y1) - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    small_frame = cv.resize(frame, (640, 480))  

  
    cv.imshow('Video', small_frame)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

cap.release()
cv.destroyAllWindows()

