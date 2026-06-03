# Driver Drowsiness Detection - Setup & Run Guide

## Recommended Runtime
- Python 3.11
- Webcam
- A locally created virtual environment

The checked-in `.venv` may point to a different machine or Python install, so recreate it locally if startup fails.

## Fastest Working Path
This repository already includes a trained sklearn model in `DDDS_CNN/models`, so you can run detection without downloading a dataset or retraining first.

From the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install opencv-python pygame numpy joblib scikit-learn
.\.venv\Scripts\python.exe .\Driver-Drowsiness-Detection\DDDS_CNN\main_capture_sklearn.py
```

Press `q` in the webcam window to stop the program.

## Web Frontend
If you want a browser-based interface with a live animated dashboard:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r .\Driver-Drowsiness-Detection\requirements.txt
.\.venv\Scripts\python.exe .\Driver-Drowsiness-Detection\app.py
```

Then open `http://127.0.0.1:5000` in your browser and click `Start Camera Detection`.

## Other Options
### Transfer Learning
1. Download the dataset from `Transfer_learning/get_dataset.txt`.
2. Place the dataset in the expected folder.
3. Run `Transfer_learning/Model_Training_TL.ipynb`.
4. Run `Transfer_learning/tl_capture.py`.

### Original CNN Training Flow
1. Download the dataset from `DDDS_CNN/Dataset.txt`.
2. Place the dataset in the expected folder.
3. Run `DDDS_CNN/model_training.py`.
4. Run `DDDS_CNN/main_capture.py`.

## What You Should See
- A live webcam window
- Eye boxes labeled `Open` or `Closed`
- A drowsiness score
- An alarm when the score stays high long enough
