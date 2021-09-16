import os
from PIL import Image
folder1 = r"C:\Users\AnKy\Downloads\Validation\male"

files = os.listdir(folder1)

size = (64,64)
a = 0

for x in files:
    a +=1
    img = Image.open("{}\{}".format(folder1,x))
    print("Resizing...")
    r_img = img.resize(size)
    print("Saving..")
    r_img.save(r"C:\Users\AnKy\Downloads\gender_disc\test\male\{}.jpg".format(a))
    

