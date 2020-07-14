import os
import cv2 as cv
import numpy as np



APPEARED_LETTERS = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
    'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
]
CAT2CHR = dict(zip(range(len(APPEARED_LETTERS)), APPEARED_LETTERS))
CHR2CAT = dict(zip(APPEARED_LETTERS, range(len(APPEARED_LETTERS))))


def distinct_char(folder):
    chars = set()
    for fn in os.listdir(folder):
        if fn.endswith('.jpg'):
            for letter in fn.split('.')[0]:
                chars.add(letter)
    return sorted(list(chars))


def load_data(folder):

    img_list = [i for i in os.listdir(folder) if i.endswith('jpg')]
    # list = Index.getIndex()
    # # print(list)
    letters_num = len(img_list)*9
    print('total letters:', letters_num)
    data = np.empty((letters_num, 60, 27,3), dtype="uint8")  # channel last
    label = np.empty((letters_num,))
    for index, img_name in enumerate(img_list):
        path1 = os.path.join(folder, img_name)
        img = cv.imread(path1)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        #轮廓检测
        contours, hierarchy = cv.findContours(binary, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        bg = []
        for i in range(len(contours)):
            bg.append(cv.boundingRect(contours[i]))
        bg.sort(key=lambda x: x[0])
        bg.remove(bg[0])
        a = []
        b = []
        #根据检测到轮廓的进行筛选，可自调
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
        for sub_index, imglc in enumerate(bg):
            x, y, w, h = imglc
            try:
                newimage=cv.resize(img[y-10:y+h,x:x+w],(27,60)) # 先用y确定高，再用x确定宽
                # newimage = np.expand_dims(newimage, axis=2)

            except:
                newimage = cv.resize(img[y:y + h, x:x + w], (27, 60))
                #newimage= np.expand_dims(newimage, axis=2)
            data[index * 9 + sub_index, :, :, :] = newimage/200
            label[index*9+sub_index] = CHR2CAT[img_name[sub_index]]
        if index % 100 == 0:
            print('{} letters loads'.format(index*9))
    return data, label

#可视化
if __name__ == '__main__':
    # print(distinct_char('../data'))
    d, l = load_data('../mnist')
    for n, i in enumerate(d):
        cv.imshow(CAT2CHR[l[n]], i*255)
        print(CAT2CHR[l[n]])
        cv.waitKey(0)
