from roboflow import Roboflow
rf = Roboflow(api_key="7Y7YXSS7r4V4SQiR9xiR")
project = rf.workspace().project("steel-rod-counting")
model = project.version(1).model.download()
# version.download("coco")  # Try yolov8 or onnx
result = model.predict(r"C:\Users\shop with hope\Downloads\original.jpg", confidence=40, overlap=30).json()