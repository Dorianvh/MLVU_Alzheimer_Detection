import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Use a specific backend that should work across different environments


# Define the directory containing the 'data' folder
DATA_DIRECTORY = 'data'

def read_data_from_folder(directory,plot_class_dist = False):
    # Initialize the feature vectors and labels lists
    X = []
    y = []

    # Define the labels
    lables = ['Mild_Demented', 'Non_Demented','Moderate_Demented', 'Very_Mild_Demented']

    # Walk through the data directory
    for root, dirs, files in os.walk(directory):
        label = os.path.basename(root)
        # Process only the directories that match the labels
        if label in lables:
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

    if plot_class_dist == True:
        unique_labels, counts = np.unique(y, return_counts=True)
        plt.bar(unique_labels, counts)
        plt.xlabel('Labels')
        plt.ylabel('Frequency')
        plt.title('Label Frequency')
        plt.show()

    return X, y

# This method one hot encodes the labels so the work with the NN
def onehot_encode_labels(labels_list):
    encoder = OneHotEncoder(sparse=False)  # We use sparse=False to get a dense matrix
    labels_list_encoded = encoder.fit_transform(y.reshape(-1, 1))  # Reshape y to be a 2D array
    return labels_list_encoded

def plot_variance_explained(data):
    pca = PCA()
    pca.fit(data)

    variance_explained = pca.explained_variance_ratio_
    cumulative_variance_explained = np.cumsum(variance_explained) * 100

    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_variance_explained, marker='o', linestyle='-')
    plt.title('Cumulative Variance Explained by Principal Components')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Variance Explained (%)')
    plt.grid(True)
    plt.xticks(np.arange(0, len(data[0]) + 1, step=500))  # Adjust x-axis ticks for better readability
    plt.yticks(np.arange(0, 101, step=5))  # Adjust y-axis ticks for better readability
    plt.show()

# This method will compute how many dimensions we need to maintain a specific amount of total variance and will reduce to that number
def reduce_dimensions_to_optimal_value(data, variance_threshold=0.90):
    # Initialize PCA and fit it to the data
    pca = PCA().fit(data)

    # This will give us the cumulative variance explained by the principal components
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # Find the number of components where the cumulative variance reaches the threshold
    num_components = np.argmax(cumulative_variance >= variance_threshold) + 1 # + 1 because array counting starts at 0 but number of dimensions start i

    pca = PCA(n_components=num_components)
    pca = PCA().fit(data)
    data = pca.transform(data)
    return data


X, y = read_data_from_folder(DATA_DIRECTORY,False)
y = onehot_encode_labels(y)





