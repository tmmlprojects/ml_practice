from ultralytics import YOLO
import torch

def main():
    model = YOLO("yolov8l.pt")
    imgsz = 640
    data_yaml = "dataset/data.yaml"
    max_batch = 1

    print(f"🧠 Testing batch sizes for: {model.model.args.get('model', 'yolov8l.pt')}, imgsz={imgsz}")
    while True:
        try:
            model.train(data=data_yaml, epochs=1, imgsz=imgsz, batch=max_batch, device=0)
            print(f"[✔] Batch size {max_batch} OK")
            max_batch *= 2
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[❌] Out of memory at batch size {max_batch}")
                print(f"✅ Max safe batch size: {max_batch // 2}")
                break
            else:
                raise e

if __name__ == "__main__":
    main()