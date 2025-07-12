from ultralytics import YOLO
import wandb
wandb.init(project="drone-human-detection", name="yolov8-topdown-v1")

def main():
    DATA_YAML = "dataset/data.yaml"
    model = YOLO("yolov8n.pt")
    model.train(
        data=DATA_YAML,
        epochs=30,
        imgsz=640,
        batch=32,
        patience=10,
        project="drone-human-detection",
        name="yolov8-topdown-v1",
        tracker="wandb"
    )

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
