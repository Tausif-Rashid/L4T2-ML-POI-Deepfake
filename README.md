# Audio-Only POI-Forensics

This codebase has been modified to focus exclusively on Person-Of-Interest (POI) deepfake detection via **audio modality**. It replaces the multimodal capability, completely omitting visual tasks (like 3DMM tracking and bounding boxes) to strictly assess the biometric validity of a speaker's voice. <br>
Kaggle Notebook: https://www.kaggle.com/code/tausifr/ml-prj-p2 <br>
Bengali Deepfake audio generation for testing : https://github.com/ibnul-nabil/CSE472_ML_Project/tree/main 

### Installation
1. Install CUDA and FFmpeg on your system.
2. Install Python>=3.7 with PyTorch>=1.7.1, TorchVision>=0.8.2, TorchAudio>=0.7.2 and Pip.
3. Install Python requirements mentioned in installs.txt
4. Download and unzip the resources folder (MD5 of zip file: d099ee4bda5b833514214c15469deadc):
```bash
    wget  https://www.grip.unina.it/download/poiforensics_resources.zip
    unzip poiforensics_resources.zip
```

### Scripts & Pipeline Usage

The testing pipeline logic is separated into the following clear phases, handled by dedicated scripts:

#### 1. Generate Voice References (`main_gen_references-copy.py`)
Extracts fundamental voice embeddings from a collection of authentic reference videos/audios associated with a POI.
- **Usage:**
```bash
export PYTHONPATH="${PYTHONPATH}:./pythonlib/"
python main_gen_references-copy.py --dir_videos "${INPUT_DIR}" --dir_poi "${POI_DIR}" --gpu 0
```
- **Results:** Generates `.npz` feature files spanning the detected audio track for each file inside the requested `${POI_DIR}`. *(Note: `main_feat_extractor-copy.py` acts as an underlying tool specifically invoked by the references engine to decouple audio spectogram loading logic from video/frame mechanisms).*

#### 2. Single Target Testing (`main_test-copy.py`)
Computes similarity distances and a deepfake global score for a single input audio/video file when compared against the POI reference embeddings.
- **Usage:**
```bash
export PYTHONPATH="${PYTHONPATH}:./pythonlib/"
python main_test-copy.py --file_video_input "${INPUT_AUDIO}" --dir_poi "${POI_DIR}" \
                         --file_output "${OUTPUT_NPZ}" --modality onlyaudio \
                         --create_plot 1 --gpu 0
```
- **Results:** Outputs an `.npz` array file with `global_score` and distance sequences. The flag `--create_plot 1` drops a qualitative chart (PNG) plotting similarity distances into the same folder.

#### 3. Automated Batch Testing (`run_tests_batch.py`)
Sequentially runs testing across an entire dataset composed of `.wav` files (segregated by `fake/` and `real/`).
- **Usage:** Verify inline paths, then execute `python run_tests_batch.py` directly.
- **Results:** Iteratively parses every record and aggregates their respective `.npz` and `.png` outputs into the specified testing output directories.

#### 4. Threshold Calibration & Dynamic Prediction (`make_prediction.py`)
Dynamically learns an optimal authentication threshold by assessing a subset of the dataset's `global_score` distributions (e.g., F1 Optimizing, 90th percentile of True Reals, or Equal Error Rate), applying it universally against test sections.
- **Usage:** Customize directories and threshold technique (e.g. `method = 'f1_optimized'`) in the file, then run `python make_prediction.py`.
- **Results:** Prints confusion matrices with Precision, Accuracy, Recall, and F1. Bundles tests into `results_prediction-{method}.txt` and groups all evaluated dataset scores into an `all_scores_dataset.csv`.

#### 5. Fixed Set Evaluation (`evaluate_results.py`)
Executes a quick check iteratively predicting outcomes using a hard-coded strict threshold rating (standardized predominantly at `score >= 0.2` implies fake).
- **Usage:** Adjust inline paths, then run `python evaluate_results.py`.
- **Results:** Asserts true/false metrics directly dumping findings to a formatted `.txt` report.


### License
Copyright (c) 2023 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').

All rights reserved.

This software should be used, reproduced and modified only for informational and nonprofit purposes.

By downloading and/or using any of these files, you implicitly agree to all the
terms of the license, as specified in the document LICENSE.txt
(included in this package) 

### References
- [1] D. Cozzolino, A. Pianese, M. Nießner, L. Verdoliva “Audio-Visual Person-of-Interest Deepfake Detection” CVPR workshop 2023
- [2] D. Cozzolino, A. Rössler, J. Thies, M. Nießner, L. Verdoliva “ID-Reveal: Identity-aware Deepfake video Detection” ICCV 2021
