import numpy as np 
 
# Load embeddings from a .npz file 
# data = np.load('/home/tr/MEGA/programming2/L4T2/ML_prj_part2_v2/pois/nicolas-cage/app_poiforensics/vid__4PKe8WGCPg/embs_track16.npz') 
data = np.load('/home/tr/MEGA/programming2/L4T2/ML_prj_part2_v2/output/testV1.npz')
dict_out = dict(data) 

print(dict_out)
 
# Access specific embeddings 
audio_embeddings = dict_out['embs_feat_audio']  # Shape: (N, feature_dim) 
video_embeddings = dict_out['embs_feat_video']  # Shape: (N, feature_dim) 
# track_ids = dict_out['embs_track']              # Which person each embedding belongs to 
# distances = dict_out['embs_dists']              # Distance to reference

print("Audio Embeddings Shape:", audio_embeddings.shape)
print("Video Embeddings Shape:", video_embeddings.shape)

print(audio_embeddings)
