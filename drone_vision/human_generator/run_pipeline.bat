@echo off
setlocal

:: Timestamp for log folders
set "TIMESTAMP=%date:~10,4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%%time:~6,2%"
set "LOGDIR=logs\run_%TIMESTAMP%"
mkdir %LOGDIR%

echo [0/6] 🧼 Cleanup (except training images)...
python cleanup_for_retrain.py > %LOGDIR%\0_cleanup.log

echo [1/6] 🤖 Auto-labeling with YOLOv8...
python auto_label_yolov8.py > %LOGDIR%\1_autolabel.log

echo [2/6] 🪓 Splitting dataset into train/val/test...
python split_dataset.py > %LOGDIR%\2_split.log

echo [3/6] 🚀 Launching YOLOv8 training...
python train_yolov8.py > %LOGDIR%\3_train_yolov8.log

echo [4/6] 🎯 Training LoRA model...
python train_lora_topdown.py > %LOGDIR%\4_train_lora.log

echo [5/6] 🔍 Inference using trained LoRA...
python inference_lora_topdown.py > %LOGDIR%\5_inference.log

echo ✅ All steps completed and logs saved in %LOGDIR%
pause
