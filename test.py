import time
import cv2
import torch
from ultralytics import YOLO

def main(
    weights: str = 'runs/train/py_exp11/weights/best.pt',
    source: int = 0,           # webcam index
    imgsz: int = 640,          # inference size
    device: str = 'cuda',      # or 'cpu'
    half: bool = True          # use FP16 if on CUDA
):
    # 1. Initialize
    model = YOLO("yolo11n.pt")
    model = YOLO(weights)
    model.fuse()  # fuse Conv+BN for speed, especially on GPU
    model.to(device)
    if device.startswith('cuda') and half:
        model.model.half()  # convert to FP16

    # 2. Open webcam
    cap = cv2.VideoCapture(source)
    assert cap.isOpened(), f'Unable to open camera {source}'
    CAM_W = 1280#1920
    CAM_H = 720#1080
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    # (Optional) Warm up
    _ = model(cv2.resize(cv2.UMat(640, 640, cv2.CV_8UC3).get(), (imgsz, imgsz)))

    prev_time = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 3. Pre-process & inference
            img = cv2.resize(frame, (imgsz, imgsz))
            # Ultraly tics automatically handles preprocessing (letterbox, normalization, etc.)
            results = model(frame, device=device, half=half, conf=0.20, iou=0.30)[0]

            # 4. Render detections
            annotated = results.plot()  # returns a numpy array with boxes & labels

            # 5. FPS counter
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time else 0
            prev_time = curr_time
            #cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30),
            cv2.putText(annotated, f'FPS: {fps:.1f}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 6. Display
            cv2.imshow('YOLOv11 Detection', annotated)
            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):  # ESC or q to quit
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

