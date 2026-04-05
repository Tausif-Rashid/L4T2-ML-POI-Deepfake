export PYTHONPATH="${PYTHONPATH}:./pythonlib/"
mkdir -p ./nicolas-cage

for id in '_4PKe8WGCPg' 'caxMBk1__-Y' 'GdxofSvTYUI' 'M0iV5vIABX0' 'TccwMWVtmj0'
do
echo $id
# DOWNLOAD video from youtube
yt-dlp --format 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best' --hls-prefer-ffmpeg --output "./nicolas-cage/vid_%(id)s.%(ext)s" --verbose -- "${id}"
done

# Create POI directory
python main_gen_references.py --dir_videos ./nicolas-cage --dir_poi ./nicolas-cage --gpu 0

#NOTE: The generated directory has to be cleaned deleting the files relative to tracks that are not of the POI.
#The good tracks should be:
#  'vid__4PKe8WGCPg': [16,22,24,27,33,36,39,45,47,48,51,55,58,68,70,75,78,81,84,88,95,98,127,130,133,136,139,146,149,152]
#  'vid_caxMBk1__-Y': [21,30,31,45,56,58,59,60]
#  'vid_GdxofSvTYUI': [17,21,25,29,31,36]
#  'vid_M0iV5vIABX0': [2,4,6,8,10],
#  'vid_TccwMWVtmj0': [0,1,3,4,6,7,9,10,11,12,14,16,20,21,26,27,28,30,32,34,35,37,39,38,43,44]
#