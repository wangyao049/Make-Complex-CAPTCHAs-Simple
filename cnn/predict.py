import numpy as np
from core import train, utils
import os
import cv2 as cv
# model = train.build_model()


def predict(raw_img, model_path):
    model = train.build_model()
    # list = Index.getIndex(raw_img,num)
    model.load_weights(model_path)
    gray = cv.cvtColor(raw_img, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    contours, hierarchy = cv.findContours(binary, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    bg = []
    for i in range(len(contours)):
        bg.append(cv.boundingRect(contours[i]))
    bg.sort(key=lambda x: x[0])
    bg.remove(bg[0])
    a = []
    b = []
    for cgindex, i in enumerate(bg):
        if i[2] + i[3] < 33:
            a.append(cgindex)
        if i[2] > 38:
            a.append(cgindex)
            b.append((i[0], i[1], i[2] // 2, i[3]))
            b.append((i[0] + i[2] // 2, i[1], i[2] // 2 + 1, i[3]))
    bg = [bg[i] for i in range(len(bg)) if (i not in a)]
    for ex in b:
        bg.append(ex)
    bg.sort(key=lambda x: x[0])
    data = np.empty((len(bg), 60, 27, 3), dtype="uint8")
    for sub_index, imglc in enumerate(bg):
        x, y, w, h = imglc
        try:
            newimage = cv.resize(raw_img[y - 10:y + h, x:x + w], (27, 60))  # 先用y确定高，再用x确定宽

        except:
            newimage = cv.resize(raw_img[y:y + h, x:x + w], (27, 60))
        data[sub_index, :, :, :] = newimage/ 200
    out = model.predict(data)
    result = np.array([np.argmax(i) for i in out])

    return ''.join([utils.CAT2CHR[i] for i in result])


if __name__ == '__main__':
    path="../mnist/"
    a=0
    for path, dir_list, file_list in os.walk(path):
        for i,file_name in enumerate(file_list):
            path1 = os.path.join(path, file_name)
            img=cv.imread(path1)
            answer = predict(img,'../model/50.hdf5')
            print(answer)
    #         name=file_name.split(".")[0]
    #         if name==answer:
    #             a=a+1
    #         else:
    #             print(name+"."+answer)
    #         # print(a)
    # s=a/1423
    # print(s)
