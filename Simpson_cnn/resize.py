import os
from PIL import Image

folder1 = r"/home/anky/Downloads/cars_data/cars_test/cars_test"

files = os.listdir(folder1)

size = (256,256)
a = 8145

for x in files:
    a +=1
    img = Image.open("{}/{}".format(folder1,x))
    print("Resizing:",a)
    r_img = img.resize(size)
    print("Saving:",a)
    r_img.save(r"/home/anky/Documents/GitHub/Machine-learning/cars_gan/cars_data/images/{}.jpg".format(a))
