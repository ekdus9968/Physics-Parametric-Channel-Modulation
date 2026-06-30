from ultralytics import YOLO

def main():
    model = YOLO("yolov8m.pt")

    model.train(
        data="data/ruod.yaml",
        epochs=12,
        imgsz=640,
        batch=8,
        project="work_dirs",
        name="yolo_ruod_baseline",
        device=0,
    )

if __name__ == "__main__":
    main()