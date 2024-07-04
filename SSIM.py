import cv2
import os
import wandb
from skimage.metrics import structural_similarity as ssim

wandb.init(project='ssim_project',name = "Whole Noisy")

def compare_images(path1, path2):
    imageA = cv2.imread(path1)
    imageB = cv2.imread(path2)

    # Compute the SSIM between the two images, specifying the channel axis aka RGB stuff
    score, _ = ssim(imageA, imageB, channel_axis=-1, full=True)
    return score

# Directories containing images
folder1 = r'C:\Users\Arvind Natarajan\Desktop\Odyssey\OOD Molecules\Noisy Predicted'
folder2 = r'C:\Users\Arvind Natarajan\Desktop\Odyssey\OOD Molecules\Ground Truth'


# Get list of image paths
images1 = [os.path.join(folder1, f) for f in os.listdir(folder1) if f.endswith(('.png'))]
images2 = {os.path.basename(f): os.path.join(folder2, f) for f in os.listdir(folder2) if f.endswith(('.png'))}

# Calculate SSIM for each pair and log to WandB
for img_path1 in images1:
    img_name = os.path.basename(img_path1)
    if img_name in images2:
        ssim_score = compare_images(img_path1, images2[img_name])
        
        # Log details including the image name and the SSIM value
        wandb.log({
            'Image Name': img_name,
            'SSIM': ssim_score,
            'image1': wandb.Image(img_path1),  # Log the path of the first image
            'image2': wandb.Image(images2[img_name])  # Log the path of the second image
        })

# Close the WandB run
wandb.finish()

print("All SSIM values computed and logged.")
