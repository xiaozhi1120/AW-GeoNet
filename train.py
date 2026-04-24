import warnings
import sys
import os
from ultralytics import YOLO
from ultralytics.nn import tasks
sys.path.append(os.getcwd())
from models.modules import GAMSBlock, GPDA, DPSConv
tasks.GAMSBlock = GAMSBlock
tasks.GPDA = GPDA
tasks.DPSConv = DPSConv
warnings.filterwarnings('ignore')
def main():
    model = model('configs/AW-GeoNet.yaml')

    try:
        model.load('yolov8s.pt')
    except Exception:
        pass
    model.train(
        data='data/data.yaml',
        project='runs/train',
        name='geo-yolo-paper',
        device='0',
        workers=4,

        # Hyperparameters
        epochs=400,
        batch=16,
        imgsz=640,
        patience=50,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=5,
        weight_decay=0.0005,
        label_smoothing=0.1,
        dropout=0.2,
        mosaic=0.8,
        mixup=0.2,
        copy_paste=0.2,
        close_mosaic=15,
        hsv_h=0.01,
        hsv_s=0.2,
        hsv_v=0.2,
        degrees=2.0,
        translate=0.02,
        scale=0.1,
        amp=True,
        plots=True
    )


if __name__ == '__main__':
    main()
