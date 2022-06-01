# F12021TT

Data mining project to analyse time trials data from F1 2021 Game.

Results: [F1 2021 Time Trial Analysis](https://lutrarutra.github.io/F12021TT/)

## Repository Contents

- `process_recording.py`: script to process screen recording of time trial leadeboard and save results to `.csv`-file
- `notebooks/process_times.py` functions to read and plot the times as well as to generate `circuits/*.html` file for each circuit
- `notebooks/*.ipynb`: plots

## 3rd Party
- Screen recording: [OBS](https://github.com/obsproject/obs-studio)
- Character recognition: [Tessaract](https://github.com/tesseract-ocr/tesseract)
- Process recording: [OpenCV](https://github.com/opencv/opencv)
- Python libraries: Pandas, NumPy, MatPlotLib, SeaBorn, PIL
