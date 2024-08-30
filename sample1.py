import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch
import os
from glob import glob
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd

# Cache the model loading
@st.cache_resource
def load_text_model():
    return SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')

# Cache the loading of embeddings
@st.cache_data 
def load_clip_embeddings(clip_features_folder, all_keyframe, video_keyframe_dict):
    embedding_dict = {}
    for v in video_keyframe_dict.keys():
        clip_path = os.path.join(clip_features_folder, f'{v}.npy')
        a = np.load(clip_path)
        embedding_dict[v] = {}
        for i, k in enumerate(video_keyframe_dict[v]):
            embedding_dict[v][k] = a[i]

    clip_embeddings = []
    for kf in all_keyframe:
        _, vid, kf = kf[:-4].rsplit('\\', 2)
        clip_embedding = embedding_dict[vid][kf]
        clip_embeddings.append(clip_embedding)

    return clip_embeddings

# Load the text model
text_model = load_text_model()

# Streamlit UI
st.title("Huy Tran Test Image Search")

# User input for search query
search_query = st.text_input("Enter your search query:")

# Paths and data structure preparation
clip_features_folder = 'clip-features-32-b1\\clip-features-32'
all_keyframe = glob('Keyframes_L01\\keyframes\\*\\*.jpg')
video_keyframe_dict = {}
all_video = glob('Keyframes_L01\\keyframes\\*')
all_video = [v.rsplit('\\', 1)[-1] for v in all_video]

for kf in all_keyframe:
    _, vid, kf = kf[:-4].rsplit('\\', 2)
    if vid not in video_keyframe_dict.keys():
        video_keyframe_dict[vid] = [kf]
    else:
        video_keyframe_dict[vid].append(kf)

for k, v in video_keyframe_dict.items():
    video_keyframe_dict[k] = sorted(v)

# Load precomputed embeddings
clip_embeddings = load_clip_embeddings(clip_features_folder, all_keyframe, video_keyframe_dict)

# Determine the value of k based on the number of available images
k = 10

# Generate a text embedding to check its shape
text_embedding = text_model.encode([search_query])

text_embedding = torch.tensor(text_embedding, dtype=torch.float32)

# Convert each clip embedding to the same data type (float32)
clip_embeddings = torch.tensor(np.array(clip_embeddings), dtype=torch.float32)

# Search and Display Results
if st.button("Search") and search_query and k > 0:
    text_embeddings = text_model.encode([search_query])
    cos_sim = util.cos_sim(text_embeddings, clip_embeddings)[0]

    # Get indices of the images with highest similarity scores
    top_indices = torch.topk(cos_sim, k=k).indices

    st.write(f"Top {k} Relevant Images:")
    for idx in top_indices[0:]:
        # Get video and framid
        _, vid, kf = all_keyframe[idx][:-4].rsplit('\\', 2)
        
        #Map the frameid to the actual frame_idx
        csv_file_path = os.path.join('map-keyframes-b1\\map-keyframes\\', f'{vid}.csv')
        map_keyframe = pd.read_csv(csv_file_path)
        frameid_to_idx_mapping = dict(zip(map_keyframe['n'], map_keyframe['frame_idx']))
        frame_idx = frameid_to_idx_mapping.get(int(kf))
        
        st.image(all_keyframe[idx], caption=f"Video: {vid}; Frame: {frame_idx}")