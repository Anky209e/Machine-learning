import os
from PIL import Image

folder1 = r"/home/sacreds/Downloads/lenses/train/sub"

files = os.listdir(folder1)

size = (64,64)
a = 0

for x in files:
    a +=1
    img = Image.open("{}/{}".format(folder1,x))
    print("Resizing:",a)
    r_img = img.resize(size)
    print("Saving:",a)
    r_img.save(r"/home/sacreds/Documents/GitHub/ML4SC_GSOC/Exploring Transformers/lenses/train/sub/{}.jpg".format(a))
