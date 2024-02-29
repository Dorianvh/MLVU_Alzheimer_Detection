import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA

# Define the directory containing the 'data' folder
DATA_DIRECTORY = 'data'

def read_data_from_folder(directory):
    # Initialize the feature vectors and labels lists
    X = []
    y = []

    # Define the labels
    lables = ['Mild_Demented', 'Moderate_Demented', 'Non_Demented', 'Very_Mild_Demented']

    # Walk through the data directory
    for root, dirs, files in os.walk(directory):
        label = os.path.basename(root)
        # Process only the directories that match the expected labels
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

    return X, y

# This method one hot encodes the labels so the work with the NN
def onehot_encode_labels(labels_list):
    encoder = OneHotEncoder(sparse=False)  # We use sparse=False to get a dense matrix
    labels_list_encoded = encoder.fit_transform(y.reshape(-1, 1))  # Reshape y to be a 2D array
    return labels_list_encoded




# This method will compute how many dimensions we need to maintain a specific amount of total variance and will reduce to that number
def reduce_dimensions_to_optimal_value(data, variance_threshold=0.95):
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


X, y = read_data_from_folder(DATA_DIRECTORY)
y = onehot_encode_labels(y)
X = reduce_dimensions_to_optimal_value(X)
print(X[0])
# Display the shapes of the arrays as a sanity check
print(X.shape, y.shape)

