@echo off
REM Training script for classifier
.\venv\Scripts\python.exe classifier.py --batch_size 16 --num_epochs %1 --num_workers 0
pause
