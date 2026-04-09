import numpy as np 
 
# Load embeddings from a .npz file 
# data = np.load('/home/tr/MEGA/programming2/L4T2/ML_prj_part2_v2/pois/nicolas-cage/app_poiforensics/vid__4PKe8WGCPg/embs_track16.npz') 
data = np.load('output/testV3-fk.npz')
print("The keys (arrays) inside this .npz file are:", data.files)
print("\nDetailed contents:")
for key in data.files:
    print(f"- {key}: shape {data[key].shape}")
    print(f"  Values:\n{data[key]}\n")
