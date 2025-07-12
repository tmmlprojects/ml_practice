@echo off
echo 📦 Evaluating YOLOv8 model on test set...
python evaluate_test_set.py
echo ✅ Done! Test results and prediction images are in \predictions\test
pause