from ultralytics import YOLO
import os
import cv2
import time

# Load YOLOv8 model
model = YOLO('yolov8m.pt')

# Load the image
image_path = "sample_images/1200x810.jpg"
image = cv2.imread(image_path)

# Run YOLOv8 inference

results = model.predict(image)

for result in results[0].boxes.data.tolist():
    xmin, ymin, xmax, ymax, conf, cls = result
    
    if cls == 0:
        image = cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
        image = cv2.putText(image, str(cls), (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
# Display the image with bounding boxes
cv2.imshow("Detected Objects", image)

# Press 'q' to exit
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
