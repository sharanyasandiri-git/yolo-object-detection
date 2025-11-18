# yolo-object-detection

from ultralytics import YOLO
import numpy

#load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt", "v8")  

#predict on an image
detection_output = model.predict(source=r"C:\Users\KISHOR\OneDrive\Documents\image detection\images5.jpg", conf=0.25, save=True)
#(copy the path of downloaded image and paste.the image should be present in same folder as code )

#Display tensor array
print(detection_output)

#Display numpy array
print(detection_output[0].numpy())


