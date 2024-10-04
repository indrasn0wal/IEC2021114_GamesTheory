import os
import shutil
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt 
import base64
from PIL import Image
import io
import math 
from math import sqrt
from sklearn.cluster import KMeans
import cv2
from ultralytics import YOLO

global embed
embed = hub.KerasLayer(os.getcwd())
model = YOLO('yolov8s-seg.pt')
results = model(["output.jpg"],conf=0.73)
result = results[0]
img = cv2.imread("output.jpg")
for idx,box in enumerate(result.boxes.xyxy):
    x1,y1,x2,y2 = box.cpu().numpy().astype(int)
    cv2.imwrite(f"reference_images/player{idx}.png", img[y1:y2,x1:x2,:])

class TensorVector(object):
    def __init__(self, FileName=None):
        self.FileName = FileName

    def process(self):

        img = tf.io.read_file(self.FileName)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.resize_with_pad(img, 224, 224)
        img = tf.image.convert_image_dtype(img,tf.float32)[tf.newaxis, ...]
        features = embed(img)
        feature_set = np.squeeze(features)
        return list(feature_set)

def convertBase64(FileName):
    """
    Return the Numpy array for a image 
    """
    with open(FileName, "rb") as f:
        data = f.read()
        
    res = base64.b64encode(data)
    
    base64data = res.decode("UTF-8")
    
    imgdata = base64.b64decode(base64data)
    
    image = Image.open(io.BytesIO(imgdata))
    
    return np.array(image)

def cosineSim(a1,a2):
    sum = 0
    suma1 = 0
    sumb1 = 0
    for i,j in zip(a1, a2):
        suma1 += i * i
        sumb1 += j*j
        sum += i*j
    cosine_sim = sum / ((sqrt(suma1))*(sqrt(sumb1)))
    return cosine_sim

def process_images_and_cluster():
    # 1. Extract vectors for specific reference images (player0.png, player1.png, player2.png, player3.png)
    reference_images_path = "reference_images"
    reference_files = {
        "player0.png": None,
        "player1.png": None,
        "player2.png": None,
        "player3.png": None
    }

    # Process reference images
    for player in reference_files.keys():
        image_path = os.path.join(reference_images_path, player)
        helper = TensorVector(image_path)
        reference_files[player] = helper.process()

    # 2. Extract vectors for images in "two_players_bot" and "two_players_top" folders
    image_folders = {
        "two_players_bot": ["player0.png", "player1.png"],
        "two_players_top": ["player2.png", "player3.png"]
    }

    image_vectors = []
    image_names = []
    folder_paths = []  # Store folder paths for each image

    # Process images in both folders
    for folder in image_folders.keys():
        for image_name in os.listdir(folder):
            image_path = os.path.join(folder, image_name)
            helper = TensorVector(image_path)
            vector = helper.process()
            image_vectors.append(vector)
            image_names.append(image_name)
            folder_paths.append(folder)  # Store original folder

    # Add reference vectors to the clustering input
    for ref_player in reference_files.keys():
        image_vectors.append(reference_files[ref_player])
        image_names.append(ref_player)

    # Convert to numpy array
    X = np.array(image_vectors)

    # 3. Apply KMeans clustering
    num_clusters = 4  # Number of players
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(X)

    # Get cluster labels
    cluster_labels = kmeans.labels_

    # 4. Create output folders
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_folders = [os.path.join(output_dir, f"player{i}") for i in range(4)]
    for folder in output_folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # 5. Move the images to their corresponding clusters
    for i, image_name in enumerate(image_names):
        cluster = cluster_labels[i]
        if image_name in reference_files:
            continue  # Skip moving reference images
        best_match_folder = output_folders[cluster]
        original_image_path = os.path.join(folder_paths[i], image_name)
        if os.path.exists(original_image_path):
            shutil.move(original_image_path, best_match_folder)

# Run the process when this script is executed
if __name__ == "__main__":
    process_images_and_cluster()