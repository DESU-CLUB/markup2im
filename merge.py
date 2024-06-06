import os
import cv2
import numpy as np

def merge_images(folder1, folder2, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get the list of image files in both folders
    files1 = [f for f in os.listdir(folder1) if f.endswith('.png')]
    files2 = [f for f in os.listdir(folder2) if f.endswith('.png')]

    # Sort the lists to ensure images are matched properly
    files1.sort()
    files2.sort()

    # Iterate through the files and merge corresponding images
    for file1, file2 in zip(files1, files2):
        # Load images
        img1 = cv2.imread(os.path.join(folder1, file1))
        img2 = cv2.imread(os.path.join(folder2, file2))

        # Create a black line
        height, width, _ = img1.shape
        line_thickness = 2
        black_line = np.zeros((height, line_thickness, 3), dtype=np.uint8)  # Black color

        # Concatenate images with the black line in between
        merged_img = np.hstack((img1, black_line, img2))

        # Save the merged image to the output folder
        output_file = os.path.join(output_folder, file1)
        cv2.imwrite(output_file, merged_img)

# Example usage
folder1 = r"XXX"
folder2 = r"XXX"
output_folder = r"XXX"

merge_images(folder1, folder2, output_folder)
