#!/bin/bash
# Script to convert all WAV files in the current directory to 16-bit PCM.

# Iterate over all .wav files in the current directory (non-recursive)
for wav_file in *.wav; do
    # Guard against case where no .wav files exist in the directory
    [ -e "$wav_file" ] || { echo "No .wav files found in the current directory."; exit 0; }

    echo "Processing $wav_file..."
    
    tmp_file="${wav_file}.tmp.wav"
    
    # -hide_banner -loglevel error reduces output noise
    ffmpeg -y -hide_banner -loglevel error -i "$wav_file" -acodec pcm_s16le "$tmp_file"
    
    if [ $? -eq 0 ]; then
        mv "$tmp_file" "$wav_file"
    else
        echo "Error converting $wav_file"
        rm -f "$tmp_file"
    fi
done

echo "Done!"
