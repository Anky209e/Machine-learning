import os
from PIL import Image

folder1 = r"/home/anky/Downloads/flower/all"

files = os.listdir(folder1)

size = (64,64)
a = 0

for x in files:
    a +=1
    img = Image.open("{}/{}".format(folder1,x))
    print("Resizing...")
    r_img = img.resize(size)
    print("Saving..")
    r_img.save(r"/home/anky/Documents/GitHub/Machine-learning/flower_pattern_gans/flower_data/images/{}.jpg".format(a))
