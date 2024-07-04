import lpips
import torch
from PIL import Image
from torchvision import transforms
import wandb
import os

# Initialize WandB
wandb.init(project='lpips_project')


def prepare_image(image_path):
    image = Image.open(image_path).convert('RGB') 
    transform = transforms.Compose([
        transforms.Resize((128, 128)), 
        transforms.ToTensor(),  
    ])
    return transform(image).unsqueeze(0) 

# Initialize the LPIPS model
lpips_model = lpips.LPIPS(net='alex') 
#can use vgg also alex has less conv layers

# Directories containing images
folder1 = r'C:\Users\Arvind Natarajan\Desktop\Odyssey\OOD Molecules\Noisy Predicted'
folder2 = r'C:\Users\Arvind Natarajan\Desktop\Odyssey\OOD Molecules\Ground Truth'

# Get list of image paths
images1 = [os.path.join(folder1, f) for f in os.listdir(folder1) if f.endswith(('.png'))]
images2 = [os.path.join(folder2, f) for f in os.listdir(folder2) if f.endswith(('.png'))]


# Calculate LPIPS for each pair and log to WandB
for img_path1, img_path2 in zip(images1, images2):
    image1 = prepare_image(img_path1)
    image2 = prepare_image(img_path2)
    distance = lpips_model(image1, image2)
    image_name = os.path.basename(img_path1)  
    # Log details including the image name and the LPIPS distance
    wandb.log({
        'Image Name': image_name,
        'LPIPS Distance': distance.item(),
        'image1': wandb.Image(image1.squeeze(0)),  # Log the first image
        'image2': wandb.Image(image2.squeeze(0))  # Log the second image
    })

# Close the WandB run
wandb.finish()

print("All distances computed and logged.")
