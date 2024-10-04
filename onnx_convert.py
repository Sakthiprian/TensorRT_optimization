from ultralytics import YOLO

model = YOLO('yolov8m.pt')
input_height = 640
input_width = 640
optimize_cpu = False

model.export(format="onnx", imgsz=[input_height,input_width], optimize=optimize_cpu)