# detection-model
It is a model made by yolov8 connected with llama 
which helps it to detect the environment and detect it in real time.

AS the yolo model is remotely installed its lagging a bit so there are two files name:
detection.py--> has the llama connected to the yolo model describing the video 
detection1.py--> here you can see the yolo model detecting the video and with confidence,class.

## Download YOLO Model
To run this project, you need to download the YOLO model. You can download it from the official Ultralytics repository:

```bash
wget https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov8n.pt

```
also the project contains my api_key so you will not able to access it remotely 

you can create your own going to huggingface models.
