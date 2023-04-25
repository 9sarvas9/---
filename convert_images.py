from PIL import Image
import os

# Folder path
root_path = "/home/sarvas/Documents/dataset/"

# Loop through all files in the folder
for filename in os.listdir(root_path + "Negative_1"):
    # Check if the file is an image
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        # Load the image
        img = Image.open(os.path.join(root_path + "Negative_1", filename))
        
        # Get the original size
        w, h = img.size
        
        # Compute the new size
        new_w = w * 25
        new_h = h * 25
        
        # Upscale the image using nearest-neighbor interpolation
        upscaled_img = img.resize((new_w, new_h), resample=Image.BICUBIC)
        
        # Save the upscaled image
        upscaled_img.save(os.path.join(root_path + "upscaled_negative", "upscaled_negative_" + filename))
        
        # Close the image
        img.close()
        upscaled_img.close()