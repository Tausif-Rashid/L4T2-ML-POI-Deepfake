# POI-Forensics

Test script for Person-Of-Interest (POI) forensics using both audo and video modalities.
It relies on the biometric characteristics of the POI in order to carry out video detection.

### Installation
1.	Install CUDA and FFmpeg with H264 on your system.
2.  Install Python>=3.7 with PyTorch>=1.7.1, TorchVision>=0.8.2, TorchAudio>=0.7.2 and Pip
3.	Install Python requirements executing:
```bash
    pip install -r ./requirements.txt
```
4.	Download and unzip the resources folder (MD5 of zip file: d099ee4bda5b833514214c15469deadc):
```bash
    wget  https://www.grip.unina.it/download/poiforensics_resources.zip
    unzip poiforensics_resources.zip
```

### Test on a provided POI
In the pois folder, there are already the extracted features for Nicolas Cage.
To run POI-Forensics [1], execute in a terminal the following command:
```bash
export PYTHONPATH="${PYTHONPATH}:./pythonlib/"
python main_test.py --file_video_input "${INPUT_VIDEO}" --file_output "${OUPUT_NPZ}" \
                    --dir_poi "./pois/nicolas-cage/app_poiforensics" --gpu 0 \
                    --create_plot 1 --create_videoout 1
```

where INPUT_VIDEO is the video to analyze, OUPUT_NPZ is the numpy file with results.
About other parameters:
- '--dir_poi' is features directory of POI.
- '--gpu' identifies the GPU to be used, set it to -1 if you do not want to use GPUs.
- '--create_plot' can be 0 or 1. If it is equal to 1, a png file that contains a plot of the results is created and saved in the same location of the numpy file.
- '--create_videoout' can be 0 or 1. If it is equal to 1, a video file with local scores is generated and saved in the same location of the numpy file.
- '--dist_normalization' can be 0 or 1. If it is equal to 1, the output distances are normalized using on the values obtained on pristine videos.

To run ID-Reveal [2], execute in a terminal the following command:
```bash
export PYTHONPATH="${PYTHONPATH}:./pythonlib/"
python main_test.py --file_video_input "${INPUT_VIDEO}" --file_output "${OUPUT_NPZ}" \
                    --dir_poi "./poi/nicolas-cage/app_idreveal" --gpu 0 \
                    --create_plot 1 --create_videoout 1
```


### Adding other POIs
Starting from a set of POI reference videos included in the folder INPUT_DIR, execute in a terminal:
```bash
export PYTHONPATH="${PYTHONPATH}:./pythonlib/"
python main_gen_references.py --dir_videos "${INPUT_DIR}" --dir_poi "${POI_DIR}" --gpu 0
```
where POI_DIR is the output directory. This function does the following: for each video it performs face detection (using RetinaFace) and face tracking and then for each track it computes the features.
For each reference video with filename "{videoname}.mp4", the script generates in the output directory:
1.	a video with the tracking indices with filename "track/track_{videoname}.mp4".
2.	an extracted face for each track with filename "faces/{videoname}/track_{n}.png", where n is the index of track.
3.	numpy files with the POI-Forensics features. In detail, a file is created for each detected track with filename "app_poiforensics/{videoname}/embs_track{n}.npz", where n is the index of track.
4.  numpy files with the ID-Reveal features. In detail, a file is created for each detected track with filename "app_idreveal/{videoname}/embs_track{n}.npz", where n is the index of track.

Before executing the test, the generated directory should be cleaned deleting the files relative to tracks that are not of the POI.

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
