import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
img = load_img('D:\JAFFED\jaffe\KA.AN1.39.tiff')
x = img_to_array(img)
print(x.shape)
np.random.seed(45)
path = 'D:\JAFFED\jaffe'
files = os.listdir(path)
files = files[1:]
print('picture num: ', len(files))


tag_list = ['AN', 'SA', 'SU', 'HA', 'DI', 'FE', 'NE']

def targets(filename):
    targets = []
    for f in filename:
        if tag_list[0] in f:
            targets.append(0)
        if tag_list[1] in f:
            targets.append(1)
        if tag_list[2] in f:
            targets.append(2)
        if tag_list[3] in f:
            targets.append(3)
        if tag_list[4] in f:
            targets.append(4)
        if tag_list[5] in f:
            targets.append(5)
        if tag_list[6] in f:
            targets.append(6)
    return np.array(targets)


def data(filename):
    train_images = []
    for f in filename:
        train_images.append(img_to_array(load_img(path+'\\'+f))[:, :, 0][:, :, None])
    return np.array(train_images)

y = targets(files)
from keras.utils import np_utils
y = np_utils.to_categorical(y)
x = data(files)

print(len(x))
print(x[0].shape)
print(y[0])
