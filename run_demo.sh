DIR_POI="./pois/nicolas-cage"
DIR_OUT="./output_demo"
mkdir -p "${DIR_OUT}"
export PYTHONPATH="${PYTHONPATH}:./pythonlib/"

for id in 'oLih6bDkmqg' 'Z1JyukEGjb0'
do
INPUT_VIDEO="${DIR_OUT}/vid_${id}.mp4"
echo "${INPUT_VIDEO}"

# Download Input Video
yt-dlp --format 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best' --hls-prefer-ffmpeg --output "${INPUT_VIDEO}" --verbose -- "${id}"

# Run POI-Forensics
METHOD="poiforensics"
OUPUT_NPZ="${DIR_OUT}/vid_${id}_${METHOD}.npz"
python main_test.py --file_video_input "${INPUT_VIDEO}" --file_output "${OUPUT_NPZ}" \
                    --dir_poi "${DIR_POI}/app_${METHOD}" --gpu 0 \
                    --create_plot 1 --create_videoout 1

# Run ID-Reveal
METHOD="idreveal"
OUPUT_NPZ="${DIR_OUT}/vid_${id}_${METHOD}.npz"
python main_test.py --file_video_input "${INPUT_VIDEO}" --file_output "${OUPUT_NPZ}" \
                    --dir_poi "${DIR_POI}/app_${METHOD}" --gpu 0 \
                    --create_plot 1 --create_videoout 1

done
