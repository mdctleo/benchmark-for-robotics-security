import tensorflow_datasets as tfds
import numpy as np
from PIL import Image
import random
import os

# Create output folder
os.makedirs("sampled_images", exist_ok=True)

# Load dataset
ds = tfds.load("droid_100", data_dir="gs://gresearch/robotics", split="train")

# Shuffle episodes and take first 100
episodes = list(ds.shuffle(1000, seed=42))[:100]

for i, episode in enumerate(episodes):
    steps = list(episode["steps"])
    step = random.choice(steps)  # pick one random step

    # Concatenate images side by side
    img = np.concatenate((
        step["observation"]["exterior_image_1_left"].numpy(),
        step["observation"]["exterior_image_2_left"].numpy(),
        step["observation"]["wrist_image_left"].numpy(),
    ), axis=1)

    Image.fromarray(img).save(f"external_datasets/DROID/sampled_images/frame_{i}.png")

print("Saved 1 random image from each of 100 episodes.")