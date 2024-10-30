from ultralytics import YOLO
import cv2 as cv
import requests
from dotenv import load_dotenv
import os


load_dotenv()


model = YOLO('yolov8n.pt')


cap = cv.VideoCapture('videos\\CCTV Footage.mp4')  


api_url = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-1B"


api_key = os.getenv('api_key')

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

while cap.isOpened():
    ret, frame = cap.read() 
    if not ret:
        break

   
    results = model(frame)

    for result in results:
        for obj in result.boxes.data:  
            x1, y1, x2, y2, conf, cls = obj.tolist() 

           
            class_name = model.names[int(cls)]

            
            prompt = f"tell me what you understand looking at and dexcribe about it: {class_name}."

            
            data = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 50  
                }
            }

            
            response = requests.post(api_url, headers=headers, json=data)

           
            if response.status_code == 200:

                description = response.json()[0]["generated_text"]
                print(f"Detected: {class_name} - Description: {description}")
            else:
                print(f"Error: {response.status_code}, {response.text}")

    # Resize the frame for display
    frame_resized = cv.resize(frame, (640, 480)) 

   
    cv.imshow('Detections', frame_resized)

    # Break the loop on 'q' key press
    if cv.waitKey(10) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv.destroyAllWindows()
