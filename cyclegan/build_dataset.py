import os
import sys
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
def _image_preprocessing(filename, xsize, ysize):
    im = Image.open(filename)
    if im.mode != 'RGB':
        print('Mode: ', im.mode)
        tmp = im.convert('RGB')
        im.close()
        im = tmp

    downsampled_im = ImageOps.fit(im, (xsize, ysize), method=Image.LANCZOS)
    norm_im = np.array(downsampled_im, dtype=np.float32)
    # im.save('F:/XL/IMG1/iter_%s.jpg' % filename.split('.')[0][-5:])
    downsampled_im.close()
    im.close()
    return norm_im

if __name__ == '__main__':
    pathA = 'dataset_java'#fake
    pathB = 'dataset_real'#real
    namesA = []
    namesB = []

    for name in os.listdir(pathA):
        namesA.append(os.path.join(pathA, name))

    for name in os.listdir(pathB):
        namesB.append(os.path.join(pathB, name))

    dataset_A = np.zeros((len(namesA), 60, 250,3))
    dataset_B = np.zeros((len(namesB), 60, 250,3))

    for i in range(len(namesB)):
        dataset_B[i] = _image_preprocessing(namesB[i], 250, 60)
        print(namesB[i])
    np.save('dataset_real.npy', dataset_B)
    for i in range(len(namesA)):
        dataset_A[i] = _image_preprocessing(namesA[i], 250, 60)
        print(namesA[i])
    np.save('dataset_java.npy'  , dataset_A)

