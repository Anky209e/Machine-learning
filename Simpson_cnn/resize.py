import os
from PIL import Image
from tqdm.notebook import tqdm
folder1 = r"C:\Users\AnKy\Downloads\flower_data\flowers_imitation"

files = os.listdir(folder1)

size = (256,256)
a = 0

for x in tqdm(files):
    a +=1
    img = Image.open("{}\{}".format(folder1,x))
    print("Resizing...")
    r_img = img.resize(size)
    print("Saving..")
    r_img.save(r"C:\Users\AnKy\Downloads\flower_pattern\flowers\{}.jpg".format(a))