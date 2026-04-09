import os
import subprocess
import sys
from pathlib import Path

# Base directories
base_dir = "/home/tr/MEGA/programming2/L4T2/ML_prj_part2_v2"
poi_dir = os.path.join(base_dir, "pois/agro2/app_poiforensics/")
test_script = os.path.join(base_dir, "main_test-copy.py")

data_dir = os.path.join(base_dir, "Testing/yt1/data")
output_dir = os.path.join(base_dir, "Testing/yt1/output")

# Categories
categories = ["fake", "real"]

def run_test(input_file, output_file):
    cmd = [
        sys.executable, test_script,
        "--file_video_input", str(input_file),
        "--dir_poi", str(poi_dir),
        "--file_output", str(output_file),
        "--modality", "onlyaudio",
        "--create_plot", "1",
        "--create_videoout", "0",
        "--gpu", "-1"
    ]
    
    print(f"==================================================")
    print(f"Processing: {input_file.name}")
    try:
        # Run the command
        subprocess.run(cmd, check=True)
        print(f"Successfully processed {input_file.name}")
    except subprocess.CalledProcessError as e:
        print(f"Error processing {input_file.name}: {e}")
    print(f"==================================================\n")

def main():
    for category in categories:
        in_cat_dir = Path(data_dir) / category
        out_cat_dir = Path(output_dir) / category
        
        # Ensure output directory exists
        out_cat_dir.mkdir(parents=True, exist_ok=True)
        
        if not in_cat_dir.exists():
            print(f"Directory {in_cat_dir} does not exist. Skipping.")
            continue
            
        # Find all wav files in the category directory
        wav_files = list(in_cat_dir.glob("*.wav"))
        print(f"Found {len(wav_files)} files in {in_cat_dir}")
        
        for wav_file in wav_files:
            # The output npz file
            output_file = out_cat_dir / f"{wav_file.stem}.npz"
            
            # Skip if output already exists (optional, but good for resuming)
            # if output_file.exists():
            #     print(f"Output for {wav_file.name} already exists. Skipping.")
            #     continue
                
            run_test(wav_file, output_file)

if __name__ == "__main__":
    main()
