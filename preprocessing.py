import os
import numpy as np
from PIL import Image

# Define the directory containing the 'data' folder
# Make sure to adjust this path to match your local directory structure
data_dir = 'data'

# Initialize the feature vectors and labels lists
X = []
y = []

# Define the expected labels
expected_labels = ['Mild_Demented', 'Moderate_Demented', 'Non_Demented', 'Very_Mild_Demented']

# Walk through the data directory
for root, dirs, files in os.walk(data_dir):
    label = os.path.basename(root)
    # Process only the directories that match the expected labels
    if label in expected_labels:
        print(f"Entering directory: {root}")
        for file in files:
            # Check if the file is an image
            if file.lower().endswith(('.jpg')):
                # Construct the full file path
                file_path = os.path.join(root, file)
                # Load the image
                image = Image.open(file_path)
                # Convert the image to grayscale
                image = image.convert('L')
                # Flatten the image to a 1-dimensional array
                flattened_image = np.array(image).flatten()
                # Append the flattened image to the feature vectors
                X.append(flattened_image)
                # Append the label to the labels list
                y.append(label)

# Convert the lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Display the shapes of the arrays as a sanity check
print(X.shape, y.shape)