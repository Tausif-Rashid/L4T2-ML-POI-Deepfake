#!/bin/bash

# Get all wav files into an array (sorted numerically/alphabetically)
files=( *.wav )
chunk_size=10
total_files=${#files[@]}
output_count=0

# Loop through files in steps of 10
for (( i=0; i<$total_files; i+=$chunk_size )); do
    
    # Create a temporary list file for ffmpeg concat
    list_file="list.txt"
    > "$list_file"
    
    # Add 10 files to the list
    for (( j=i; j<i+$chunk_size && j<$total_files; j++ )); do
        echo "file '${files[$j]}'" >> "$list_file"
    done
    
    # Format the output name (e.g., j_000.wav, j_001.wav)
    output_name=$(printf "j_%03d.wav" $output_count)
    
    echo "Creating $output_name..."
    
    # Run ffmpeg concat
    ffmpeg -f concat -safe 0 -i "$list_file" -c copy "$output_name" -loglevel error
    
    ((output_count++))
done

# Clean up
rm "$list_file"
echo "Done! Generated $output_count files."