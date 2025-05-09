from ultralytics import YOLO

# 1) Load pretrained model
model = YOLO('yolo11n.pt')

# 2) Train
model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    freeze=[10],
    project='runs/train',
    name='py_exp',
    half=True
)

# 3) Validate & inspect metrics
metrics = model.val()
print(f"mAP@0.5 = {metrics.box.map50:.3f}")

# 4) Export to ONNX/TF if needed
model.export(format='onnx')          # produces best.onnx
model.export(format='torchscript')   # produces best.torchscript

