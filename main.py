# Jude Tear
# Friday November 17th 2023
# CISC 471: Main

import cv2 as cv
import numpy as np
import os

from sklearn.model_selection import train_test_split

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
def readImage(path, color_format='rgb', resize_dim=None, clip_range=[-125,275]):

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
        image = cv.resize(image, resize_dim)

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

    for category in categories:
        category_input_path = os.path.join(input_dir, category)
        category_output_path = os.path.join(output_dir, category)

        if not os.path.isdir(category_input_path):
            print(f"Category input path does not exist: {category_input_path}")
            continue

        if not os.path.exists(category_output_path):
            os.makedirs(category_output_path)

        for file_name in os.listdir(category_input_path):
            if file_name.endswith('.png') and not file_name.endswith('_mask.png'):
                file_path = os.path.join(category_input_path, file_name)
                
                # Read and preprocess the image
                image = readImage(file_path, color_format='rgb')
                
                # Saving the image as a numpy array
                npy_path = os.path.join(category_output_path, file_name.replace('.png', '.npy'))
                np.save(npy_path, image)

                # Optionally, also process and save the corresponding mask
                mask_path = os.path.join(category_input_path, file_name.split('.')[0] + '_mask.png')
                if os.path.exists(mask_path):
                    mask = readImage(mask_path, color_format='gray')
                    npy_mask_path = os.path.join(category_output_path, file_name.split('.')[0] + '_mask.npy')
                    np.save(npy_mask_path, mask)



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

    for filename in sorted_filenames:
        if filename.endswith("mask.npy"):
            # Find the corresponding image file
            image_filename = filename.replace("_mask.npy", ".npy")
            image_path = os.path.join(folder, image_filename)

            if os.path.exists(image_path):
                # Load the image and mask
                image = np.load(image_path)
                mask = np.load(os.path.join(folder, filename))

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
    # convert_and_save_images(root_dir, processed_dir, categories) 


    # Load NumPy Arrays
    processed_data_dir = '/Users/judetear/Documents/CISC471/Project/Processed_Data/'
    benign_dir = os.path.join(processed_data_dir, "benign")
    malignant_dir = os.path.join(processed_data_dir, "malignant")

    benign_images, benign_masks = load_data_from_folder(benign_dir)
    malignant_images, malignant_masks = load_data_from_folder(malignant_dir)
    

    # Split BENIGN into train / test / validate
    (train_images_ben, train_masks_ben),\
    (val_images_ben, val_masks_ben),\
    (test_images_ben, test_masks_ben) = split_data(benign_images, benign_masks)

    # Check the distribution
    print(f"\nBenign dataset - Total Images = {len(benign_images)}")
    print(f"Training set: {len(train_images_ben)} images")
    print(f"Training set: {len(train_images_ben)} masks\n")

    print(f"Validation set: {len(val_images_ben)} images")
    print(f"Validation set: {len(val_images_ben)} masks\n")

    print(f"Testing set: {len(test_images_ben)} images")
    print(f"Testing set: {len(test_images_ben)} masks")
    
    # Split MALIGNANT into train / test / validate
    (train_images_mal, train_masks),\
    (val_images_mal, val_masks_mal),\
    (test_images_mal, test_masks_mal) = split_data(malignant_images, malignant_masks)

    # Check the distribution
    print(f"\nMalignant dataset - Total Images = {len(malignant_images)}")
    print(f"Training set: {len(train_images_mal)} images")
    print(f"Training set: {len(train_images_mal)} masks\n")

    print(f"Validation set: {len(val_images_mal)} images")
    print(f"Validation set: {len(val_images_mal)} masks\n")

    print(f"Testing set: {len(test_images_mal)} images")
    print(f"Testing set: {len(test_images_mal)} masks")

    print(test_image[0].shape)
   
