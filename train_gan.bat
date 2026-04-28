@echo off
REM Training script for DCGAN
.\venv\Scripts\python.exe train_gan.py --batch_size 8 --num_epochs %1 --num_workers 0 --save_interval 2
pause
