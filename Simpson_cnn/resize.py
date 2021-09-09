import os
from PIL import Image
folder1 = r"C:\Users\AnKy\Downloads\archive\simpsons_dataset\cbart_simpson"

files = os.listdir(folder1)

size = (64,64)
a = 0

for x in files:
    a +=1
    img = Image.open("{}\{}".format(folder1,x))
    print("Resizing...")
    r_img = img.resize(size)
    print("Saving..")
    r_img.save(r"C:\Programming\Machine-learning\Simpson_cnn\train_set\cbart\{}.jpg".format(a))
    

