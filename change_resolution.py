from PIL import Image


with Image.open('data/Resized/1_small.jpg') as image:
    width, height = image.size
    new_size = (int(width/2), int(height/2))
    resized_image = image.resize(new_size)
    resized_image.save('data/Resized/1__very_small.jpg')