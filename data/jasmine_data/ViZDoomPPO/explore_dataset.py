import matplotlib.pyplot as plt
import base64
import io
from PIL import Image
import numpy as np
from matplotlib.animation import FuncAnimation
import pyarrow.parquet as pq


# Example script to open the parquet file and get the data
parquet_path = "./concatenated.parquet"
table = pq.read_table(parquet_path)
# Convert to pandas DataFrame
df = table.to_pandas()

# Sort by step_id
df_sorted = df.sort_values("step_id")

# Get the first row
first_row = df_sorted.iloc[27]


# Function to decode base64 image
def decode_image(b64_string):
    image_data = base64.b64decode(b64_string)
    return np.array(Image.open(io.BytesIO(image_data)))


# Decode all images in the first row
images = [decode_image(img) for img in first_row["images"]]

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 8))
plt.axis("off")


# Animation function
def animate(i):
    ax.clear()
    ax.imshow(images[i])
    ax.set_title(
        f"Episode {first_row['episode_id']}, Step {first_row['step_id']}, Frame {i+1}/{len(images)}"
    )
    ax.axis("off")


# Create animation
anim = FuncAnimation(fig, animate, frames=len(images), interval=200, repeat=True)

plt.show()

# Print other information about the first row
print(f"Episode ID: {first_row['episode_id']}")
print(f"Step ID: {first_row['step_id']}")
print(f"Health: {first_row['health']}")
print(f"Actions: {first_row['actions']}")
