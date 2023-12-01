# Jude Tear
# Friday November 17th 2023
# CISC 471: Main

'''
Description:
Its important to note the desired folder hierarchy of this project.

The Dataset_BUSI_with_GT should contain the original .png files
It should go Project > Dataset_BUSI_with_GT > [benign/malignant/normal]

After preprocessing a new folder called "Processed_Data" should exist
containing the NumPy Arrays
It should go Project > Processed_Data > [benign/malignant/normal]

==
'''

import cv2 as cv
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from unet import unet_model

'''
Helper Function #1 - Display image
'''
def display_image_cv(filepath):
    # Load the image
    image = cv.imread(filepath)

    # Check if the image was loaded successfully
    if image is None:
        print(f"Unable to load image from {filepath}")
        return

    # Display the image in a window
    cv.imshow('Image', image)
    cv.waitKey(0)
    cv.destroyAllWindows()



'''
Helper Function #2 - Read image in
'''
def readImage(path, color_format='gray', resize_dim=(128, 128), clip_range=[-125,275]):

    # Read the image using OpenCV
    image = cv.imread(path)
    if image is None:
        raise FileNotFoundError(f"Unable to find or open the image at path: {path}")

    # Convert the color format if needed
    if color_format.lower() == 'rgb':
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    elif color_format.lower() == 'gray':
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    elif color_format.lower() != 'bgr':
        raise ValueError(f"Unsupported color format: {color_format}")
    
    # Resize if provided
    if resize_dim:

        # Differentiate between masks and images
        if color_format.lower() == 'gray':
            interpolation = cv.INTER_NEAREST  # Recommended for masks (grayscale images)
        else:
            interpolation = cv.INTER_LINEAR   # Suitable for color images

        image = cv.resize(image, resize_dim, interpolation=interpolation)

    # Apply clipping if range is provided
    if clip_range:
        image = np.clip(image, clip_range[0], clip_range[1])

    return image



'''
Helper Function #3 - Convert and Save Images
'''
def convert_and_save_images(input_dir, output_dir, categories):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize variables for tracking size extremes
    min_width, min_height = float('inf'), float('inf')
    max_width, max_height = 0, 0

    # Iterate over the categories
    for category in categories:
        category_input_path = os.path.join(input_dir, category)
        category_output_path = os.path.join(output_dir, category)

        # Ensure input and output directories exist
        if not os.path.isdir(category_input_path):
            print(f"Category input path does not exist: {category_input_path}")
            continue

        if not os.path.exists(category_output_path):
            os.makedirs(category_output_path)

        # Iterate over current categories directory 
        for file_name in os.listdir(category_input_path):

            # Get ultra sound image
            if file_name.endswith('.png') and not file_name.endswith('_mask.png'):
                file_path = os.path.join(category_input_path, file_name)
                
                # Read and preprocess the image
                image = readImage(file_path, color_format='gray')
                
                # Update the size extremes 
                height, width = image.shape[:2]
                min_width, max_width = min(min_width, width), max(max_width, width)
                min_height, max_height = min(min_height, height), max(max_height, height)

                # Save the image as NumPy array 
                npy_path = os.path.join(category_output_path, file_name.replace('.png', '.npy'))
                np.save(npy_path, image)

                # Process and save the corresponding mask
                mask_path = os.path.join(category_input_path, file_name.split('.')[0] + '_mask.png')
                if os.path.exists(mask_path):
                    mask = readImage(mask_path, color_format='gray')

                    # Update the size extremes for the mask
                    mask_height, mask_width = mask.shape[:2]
                    min_width, max_width = min(min_width, mask_width), max(max_width, mask_width)
                    min_height, max_height = min(min_height, mask_height), max(max_height, mask_height)

                    npy_mask_path = os.path.join(category_output_path, file_name.split('.')[0] + '_mask.npy')
                    
                    # Save the mask as NumPy array
                    np.save(npy_mask_path, mask)

    return min_width, min_height, max_width, max_height



'''
Helper Function #4 - Load NumPy arrays into arrays 

Files:
    - Mask and image files must follow format "benign (i)" and "benign (i)_mask.npy"
'''
def load_data_from_folder(folder):

    masks = []
    images = []

    # Sort the filenames to ensure correspondence
    sorted_filenames = sorted(os.listdir(folder))

    # Iterate over the NumPy Arrays
    for filename in sorted_filenames:

        # Get masks
        if filename.endswith("mask.npy"):

            # Find the corresponding image file
            image_filename = filename.replace("_mask.npy", ".npy")
            image_path = os.path.join(folder, image_filename)

            # If the image exists
            if os.path.exists(image_path):

                # Load the image and mask
                image = np.load(image_path)
                mask = np.load(os.path.join(folder, filename))

                # Ensure the image and mask are three-dimensional
                if len(image.shape) == 2:
                    image = image[..., tf.newaxis]
                if len(mask.shape) == 2:
                    mask = mask[..., tf.newaxis]

                # Normalize images and augment (rotate)
                image, mask = normalize(image, mask)
                image, mask = random_rotate_image_and_mask(image, mask)

                # Append to respective lists
                images.append(image)
                masks.append(mask)
            else:
                print(f"No corresponding image found for mask: {filename}")

    return images, masks



'''
Helper Function #5 - Split data into training, validation, and testing sets
'''
def split_data(images, masks, train_size=0.7, val_size=0.15, test_size=0.15):
    # Pair images with their corresponding masks
    paired_data = list(zip(images, masks))

    # Split the paired data into training and (validation + test)
    train_data, val_test_data = train_test_split(paired_data, train_size=train_size)

    # Further split the (validation + test) data into validation and test sets
    val_size_ratio = val_size / (val_size + test_size)  # Adjust validation size for the split
    val_data, test_data = train_test_split(val_test_data, train_size=val_size_ratio)

    # Unzip the pairs back into separate lists
    train_images, train_masks = zip(*train_data)
    val_images, val_masks = zip(*val_data)
    test_images, test_masks = zip(*test_data)

    return (train_images, train_masks), (val_images, val_masks), (test_images, test_masks)



'''
Helper Function #6 - Get information about images
'''
def analyze_image_intensities(image):
    # Convert to numpy array if it's a TensorFlow tensor
    if tf.is_tensor(image):
        image = image.numpy()

    # print("Sample pixel values:\n", image[0:5, 0:5])

    # print("Min intensity:", np.min(image))
    # print("Max intensity:", np.max(image))
    # print("Mean intensity:", np.mean(image))
    # print("Standard deviation:", np.std(image))

    plt.hist(image.ravel(), bins=256, range=[0,256])
    plt.title("Pixel Intensity Distribution")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.show()



'''
Helper Function #7 - Normalize image and mask
'''
def normalize(input_image, input_mask):
    # If images are binary or categorical but in the 0-255 range
    input_image = tf.cast(input_image, tf.float32) / 255.0
    
    # If masks are binary or categorical but in the 0-255 range
    input_mask = tf.cast(input_mask, tf.float32) / 255.0  # Adjust if needed based on mask range
    return input_image, input_mask



'''
Helper Function #8 - Augment images (applies a random rotation)
'''
def random_rotate_image_and_mask(image, mask):
    # Randomly choose a rotation angle
    angles = [0, 90, 180, 270]
    angle = np.random.choice(angles)

    # Perform rotation
    image = tf.image.rot90(image, k=angle // 90)
    mask = tf.image.rot90(mask, k=angle // 90)
    return image, mask



'''
Helper Function #9 - Prep
'''
# Convert the data to tf.data.Dataset
def prepare_dataset(images, masks, batch_size):
    # Convert lists to tensors
    images = tf.convert_to_tensor(images, dtype=tf.float32)
    masks = tf.convert_to_tensor(masks, dtype=tf.float32)

    # Create a TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


'''
Helper Function #10 - Plot History
'''
def plot_training_history(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')



# Main execution
if __name__ == '__main__':

    # Project information -----------------------------
    root_dir = '/Users/judetear/Documents/CISC471/Project/Dataset_BUSI_with_GT'
    processed_dir = '/Users/judetear/Documents/CISC471/Project/Processed_Data'
    categories = ['benign', 'malignant', 'normal']
    # -------------------------------------------------------------------------

    # Test image to display with - #1
    display_image_cv(os.path.join(root_dir, "benign/benign (1).png"))

    # Create 'numpy.ndarray' array - #2
    test_image = readImage(os.path.join(root_dir, "benign/benign (2).png"))

    # Convert images to numpy arrays (save as .npy's) - #3 (Done)
    min_width, min_height, max_width, max_height = convert_and_save_images(root_dir, processed_dir, categories)
    # print("Min Width: {}, Min Height: {}, Max Width: {}, Max Height: {}".format(min_width, min_height, max_width, max_height))
 
    # Images.shape = (128, 128, 3)
    # Masks.shape =  (128, 128)

    # Load NumPy Arrays
    processed_data_dir = '/Users/judetear/Documents/CISC471/Project/Processed_Data/'
    benign_dir = os.path.join(processed_data_dir, "benign")
    malignant_dir = os.path.join(processed_data_dir, "malignant")

    benign_images, benign_masks = load_data_from_folder(benign_dir)
    malignant_images, malignant_masks = load_data_from_folder(malignant_dir)

    # Check image format
    print(benign_masks[0].dtype)
    print(benign_images[0].dtype)
    

    # Split BENIGN into train / test / validate
    (train_images_ben, train_masks_ben),\
    (val_images_ben, val_masks_ben),\
    (test_images_ben, test_masks_ben) = split_data(benign_images, benign_masks)

    # Check the distribution
    print(f"\nBenign dataset - Total Images = {len(benign_images)}")
    print(f"Training set: {len(train_images_ben)} images") # There should be 305
    print(f"Training set: {len(train_images_ben)} masks\n") # There should be 305

    print(f"Validation set: {len(val_images_ben)} images") # There should be 66
    print(f"Validation set: {len(val_images_ben)} masks\n") # There should be 66

    print(f"Testing set: {len(test_images_ben)} images") # There should be 66
    print(f"Testing set: {len(test_images_ben)} masks") # There should be 66
    
    # Split MALIGNANT into train / test / validate
    (train_images_mal, train_masks_mal),\
    (val_images_mal, val_masks_mal),\
    (test_images_mal, test_masks_mal) = split_data(malignant_images, malignant_masks)

    # Check the distribution
    print(f"\nMalignant dataset - Total Images = {len(malignant_images)}")
    print(f"Training set: {len(train_images_mal)} images") # There should be 147
    print(f"Training set: {len(train_images_mal)} masks\n") # There should be 147

    print(f"Validation set: {len(val_images_mal)} images") # There should be 31
    print(f"Validation set: {len(val_images_mal)} masks\n") # There should be 31

    print(f"Testing set: {len(test_images_mal)} images") # There should be 32
    print(f"Testing set: {len(test_images_mal)} masks") # There should be 32

    print("\n")

    print(benign_images[0].dtype) # <dtype: 'float32'>
    print(benign_masks[0].dtype) # <dtype: 'float32'>

    # Analyze the intensities of a sample image
    # analyze_image_intensities(train_images_ben[0])
    # analyze_image_intensities(train_masks_ben[0])

    # We now have: Benign Train/Validate/Test
    # And we also: Malignant Train/Validate/Test

    # Adjust batch size as needed
    batch_size = 32

    # Prepare datasets
    train_dataset_ben = prepare_dataset(train_images_ben, train_masks_ben, batch_size)
    val_dataset_ben = prepare_dataset(val_images_ben, val_masks_ben, batch_size)
    test_dataset_ben = prepare_dataset(test_images_ben, test_masks_ben, batch_size)

    train_dataset_mal = prepare_dataset(train_images_mal, train_masks_mal, batch_size)
    val_dataset_mal = prepare_dataset(val_images_mal, val_masks_mal, batch_size)
    test_dataset_mal = prepare_dataset(test_images_mal, test_masks_mal, batch_size)
    

    # Compile the model
    unet_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = unet_model.fit(
        train_dataset_ben,
        epochs=10,  # Adjust the number of epochs as needed
        validation_data=val_dataset_ben)
    
    # Plot training history
    plot_training_history(history)
    
    # Define the directory path for the predictions
    prediction_dir_ben = os.path.join('/Users/judetear/Documents/CISC471/Project', 'Predictions-Benign')
    prediction_dir_mal = os.path.join('/Users/judetear/Documents/CISC471/Project', 'Predictions-Malignant')

    # Create the directory if it doesn't exist
    if not os.path.exists(prediction_dir_ben):
        os.makedirs(prediction_dir_ben)

    # Create the directory if it doesn't exist
    if not os.path.exists(prediction_dir_mal):
        os.makedirs(prediction_dir_mal)

    # Take a batch of images from the test dataset
    for test_images, test_masks in test_dataset_ben.take(1):
        # Make predictions
        predictions = unet_model.predict(test_images)

        # Save each prediction as an image
        for i, prediction in enumerate(predictions):
            # Convert the prediction to a suitable format for saving
            prediction_image = (prediction.squeeze() * 255).astype(np.uint8)

            # Create a unique filename for each prediction
            prediction_filename = f"prediction_{i + 1}.png"

            # Save the prediction image
            cv.imwrite(os.path.join(prediction_dir_ben, prediction_filename), prediction_image)

        # Optionally, display the images, true masks, and predicted masks
        for i in range(min(len(test_images), 5)):  # Display first 5 images
            plt.figure(figsize=(10, 10))
            plt.subplot(1, 3, 1)
            plt.title("Test Image - Benign")
            plt.imshow(test_images[i].numpy().squeeze(), cmap='gray')
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.title("True Mask - Benign")
            plt.imshow(test_masks[i].numpy().squeeze(), cmap='gray')
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.title("Predicted Mask - Benign")
            plt.imshow(predictions[i].squeeze(), cmap='gray')  # Adjust as needed
            plt.axis('off')

            plt.show()

    for test_images, test_masks in test_dataset_mal.take(1):
    # Make predictions for malignant data
        predictions_mal = unet_model.predict(test_images)

        # Save each malignant prediction as an image
        for i, prediction in enumerate(predictions_mal):
            # Convert the prediction to a suitable format for saving
            prediction_image_mal = (prediction.squeeze() * 255).astype(np.uint8)

            # Create a unique filename for each prediction
            prediction_filename_mal = f"mal_prediction_{i + 1}.png"

            # Save the prediction image
            cv.imwrite(os.path.join(prediction_dir_mal, prediction_filename_mal), prediction_image_mal)

        # Optionally, display the images, true masks, and predicted masks for malignant data
        for i in range(min(len(test_images), 5)):  # Display first 5 images
            plt.figure(figsize=(10, 10))
            plt.subplot(1, 3, 1)
            plt.title("Test Image - Malignant")
            plt.imshow(test_images[i].numpy().squeeze(), cmap='gray')
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.title("True Mask - Malignant")
            plt.imshow(test_masks[i].numpy().squeeze(), cmap='gray')
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.title("Predicted Mask - Malignant")
            plt.imshow(predictions_mal[i].squeeze(), cmap='gray')  # Adjust as needed
            plt.axis('off')

            plt.show()

    # Save the model
    unet_model.save('Path-to-model')