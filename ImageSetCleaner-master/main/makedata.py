import numpy as np
import pandas as pd
import os
from PIL import Image
import cv2


def luoData(path):
    lis = os.listdir('data')
    print(lis)
    labels = []
    imgData = []
    for ele in lis:
        sample = os.listdir(path + '/' + ele)
        for i in sample:
            print(i)
            if ele == "zero_1":
                # img = cv2.imread(path + "/" + ele + '/' + i)
                # img = img.convert("")
                img = Image.open(path + "/" + ele + '/' + i)
                img = img.convert('RGB')
                img = img.resize((128, 128), Image.ANTIALIAS)
                img = np.array(img)
                img = img / 255.0
                imgData.append(img)
                labels.append(0)
            if ele == 'first':
                img = Image.open(path + "/" + ele + '/' + i)
                img = img.convert('RGB')
                img = img.resize((128, 128), Image.ANTIALIAS)
                img = np.array(img)
                img = img / 255.
                imgData.append(img)
                labels.append(1)
            if ele == 'second':
                img = Image.open(path + "/" + ele + '/' + i)
                img = img.convert('RGB')
                img = img.resize((128, 128), Image.ANTIALIAS)
                img = np.array(img)
                img = img / 255.
                imgData.append(img)
                labels.append(2)
            if ele == 'three':
                img = Image.open(path + "/" + ele + '/' + i)
                img = img.convert('RGB')
                img = img.resize((128, 128), Image.ANTIALIAS)
                img = np.array(img)
                img = img / 255.
                imgData.append(img)
                labels.append(3)
            if ele == 'four':
                img = Image.open(path + "/" + ele + '/' + i)
                img = img.convert('RGB')
                img = img.resize((128, 128), Image.ANTIALIAS)
                img = np.array(img)
                img = img / 255.
                imgData.append(img)
                labels.append(4)
    imgData = np.array(imgData)
    labels = np.array(labels)

    return imgData, labels



