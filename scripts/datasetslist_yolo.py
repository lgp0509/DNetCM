import os
from tqdm.autonotebook import tqdm
from tqdm import trange

set = "glo3"
for type in ["train","val","test"]:
    path = f'C:/yolov9/datasets/{set}/images/{type}/'  # "E:/DN_Slide/"切片地址
    dir = os.listdir(path)
    slidelist = []
    out = open(f'C:/yolov9/datasets/{set}/{type}.txt', 'w')
    for file in dir:
        slidelist.append(f'./images/{type}/' + file)
        out.write(f'./images/{type}/' + file + '\n')
    out.close()
