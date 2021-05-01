import os

import numpy as np
from PIL import Image
from sklearn.datasets import fetch_openml


data_dir = "./data/"
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

print("fetch")
mnist = fetch_openml("mnist_784", version=1, data_home="./data/", as_frame=False)

X = mnist.data
y = mnist.target

data_dir_path = "./data/img_78/"
if not os.path.exists(data_dir_path):
    os.mkdir(data_dir_path)


count7 = 0
count8 = 0
max_num = 200

for i in range(len(X)):
    print(f"i={i}")

    if (y[i] == "7") and (count7 < max_num):
        file_path = "./data/img_78/img_7_" + str(count7) + ".jpg"
        im_f = X[i].reshape(28, 28)
        pil_img_f = Image.fromarray(im_f.astype(np.uint8))
        pil_img_f = pil_img_f.resize((64, 64), Image.BICUBIC)
        pil_img_f.save(file_path)
        count7 += 1

    if (y[i] == "8") and (count8 < max_num):
        file_path = "./data/img_78/img_8_" + str(count8) + ".jpg"
        im_f = X[i].reshape(28, 28)
        pil_img_f = Image.fromarray(im_f.astype(np.uint8))
        pil_img_f = pil_img_f.resize((64, 64), Image.BICUBIC)
        pil_img_f.save(file_path)
        count8 += 1
