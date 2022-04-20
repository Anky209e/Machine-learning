import os
from PIL import Image
import sys
from tqdm import tqdm
f = sys.argv

# 0 - Filename
# 1 - Image-size-x
# 2 - Image-size-y
# 3 - Parent folder
# 4 - target directory
if len(f) > 1:
    s_x = int(f[1])
    s_y = int(f[2])

    folder1 = f[3]
    folder2 = f[4]
    files = os.listdir(folder1)

    size = (s_x,s_y)
    a = 0

    for x in tqdm(files):
        a +=1
        img = Image.open("{}/{}".format(folder1,x))
        
        r_img = img.resize(size)
        
        r_img.save(r"{}/{}.jpg".format(folder2,a))
    print("Completed")
else:
    print("Please provide arguments.\n image_x,image_y,paretntpath,targetpath")
