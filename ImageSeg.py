import math
import numpy as np
import cv2
from skimage import io
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from skimage.util import img_as_float


def calcEntropy(img):
    entropy = []

    hist = cv2.calcHist([img], [0], None, [256], [0, 255])
    total_pixel = img.shape[0] * img.shape[1]

    for item in hist:
        probability = item / total_pixel
        if probability == 0:
            en = 0
        else:
            en = -1 * probability * (np.log(probability) / np.log(2))
        entropy.append(en)

    sum_en = np.sum(entropy)
    return sum_en


def crop_img(pix, RGB_image, is_H = True):
    #分割大小刚好是图片大小的1/2
    if pix == (RGB_image.shape[0])/2 or pix == (RGB_image.shape[1])/2:
        pix = pix + 1
    if is_H == True:
        a_crop_up = RGB_image[0: pix, 0: RGB_image.shape[1]]
        a_crop_down = RGB_image[pix: RGB_image.shape[0], 0: RGB_image.shape[1]]
    else:
        a_crop_up = RGB_image[0: RGB_image.shape[0], 0: pix]
        a_crop_down = RGB_image[0: RGB_image.shape[0], pix: RGB_image.shape[1]]
    cv2.imwrite("image/after_crop_up.jpg", a_crop_up)
    cv2.imwrite("image/after_crop_down.jpg", a_crop_down)
    return a_crop_up, a_crop_down


def Icr(pix, RGB_image, hist_all, Hc, is_H = True):
    # RGB_image = cv2.resize(RGB_image, (500, 373))
    if is_H == True:
        a_crop_up = RGB_image[0: pix, 0: RGB_image.shape[1]]
        a_crop_down = RGB_image[pix: RGB_image.shape[0], 0: RGB_image.shape[1]]
    else:
        a_crop_up = RGB_image[0: RGB_image.shape[0], 0: pix]
        a_crop_down = RGB_image[0: RGB_image.shape[0], pix: RGB_image.shape[1]]

    up_image = cv2.cvtColor(a_crop_up, cv2.COLOR_BGR2GRAY)
    down_image = cv2.cvtColor(a_crop_down, cv2.COLOR_BGR2GRAY)

    hist_up = cv2.calcHist([up_image], [0], None, [256], [0, 255])
    hist_down = cv2.calcHist([down_image], [0], None, [256], [0, 255])

    Hcr1 = 0
    for j in range(hist_up.shape[0]):
        if hist_all[j, 0] == 0 or hist_up[j, 0] == 0:
            continue
        Pcr = hist_up[j, 0] / np.sum(hist_up)
        Hcr = Pcr * math.log(Pcr, 2)
        Hcr1 += Hcr

    Hcr2 = 0
    for k in range(hist_down.shape[0]):
        if hist_all[k, 0] == 0 or hist_down[k, 0] == 0:
            continue
        Pcr = hist_down[k, 0] / np.sum(hist_down)
        Hcr = Pcr * math.log(Pcr, 2)
        Hcr2 += Hcr

    Hcr1 = -Hcr1
    Hcr2 = -Hcr2
    pai_1 = (a_crop_up.shape[0] * a_crop_up.shape[1]) / (RGB_image.shape[0] * RGB_image.shape[1])
    pai_2 = (a_crop_down.shape[0] * a_crop_down.shape[1]) / (RGB_image.shape[0] * RGB_image.shape[1])
    Icr = Hc[0] - (pai_1 * Hcr1 + pai_2 * Hcr2)
    return Icr


def img_process(RGB_image):
    gray_image = cv2.cvtColor(RGB_image, cv2.COLOR_BGR2GRAY)
    np_image = np.array(gray_image)
    Hc = calcEntropy(np_image)
    hist_all = cv2.calcHist([RGB_image], [0], None, [256], [0, 255])
    return RGB_image, Hc, hist_all


def information(image):
    result_img = image
    image, Hc, hist_all = img_process(image)
    # cv2.imwrite("image/ori_2.jpg", image)
    y1, x1 = [], []
    y2, x2 = [], []
    for i in range(1, image.shape[0]):
        x1.append(i)
        y1.append(Icr(i, image, hist_all, Hc, is_H=True))
    for i in range(1, image.shape[1]):
        x2.append(i)
        y2.append(Icr(i, image, hist_all, Hc, is_H=False))

    max_y1 = np.argmax(y1)
    max_y2 = np.argmax(y2)
    if y1[max_y1] > y2[max_y2]:
        w = image.shape[1]
        h = max_y1
        print("Horizontal:{}".format(h))
        # cv2.imwrite('image/result_{}.jpg'.format(1), result_img)
        seg_pix = h
        is_H = True
    else:
        w = max_y2
        h = image.shape[0]
        print("vertical:{}".format(max_y2))
        # cv2.imwrite('image/result_{}.jpg'.format(1), result_img)
        seg_pix = w
        is_H = False
    return seg_pix, is_H


def IESA(lines, w, h, im_path):
    RGB_image = cv2.imread(im_path)
    # w = 500
    # h = 373
    image = result = cv2.resize(RGB_image, (w, h))
    # 保存要处理的原图
    cv2.imwrite("image/ori.png", image)

    seg_pix, is_H = information(image)
    if seg_pix != 0:
        a_crop_up, a_crop_down = crop_img(seg_pix, image, is_H=is_H)
    else:
        if is_H == True:
            # 分割的区域不能除以2 如果除以2的话，两个区域的shape[0]和shape[1]就一样 无法判断ind的值
            a_crop_up, a_crop_down = crop_img((image.shape[0] // 2) + 1, image, is_H=is_H)
        else:
            a_crop_up, a_crop_down = crop_img((image.shape[1] // 2) + 1, image, is_H=is_H)
    # a_crop_up, a_crop_down = crop_img(seg_pix, image, is_H=is_H)
    image, mask = [], []
    image.append([a_crop_up, a_crop_down])
    mask.append([np.full((a_crop_up.shape[0], a_crop_up.shape[1]), 0),
                 np.full((a_crop_down.shape[0], a_crop_down.shape[1]), 1)])
    x = 2
    for i in range((lines-1) // 2):
        for j in range(0, 2):
            seg_pix, is_H = information(image[i][j])
            if seg_pix != 0:
                a_crop_up, a_crop_down = crop_img(seg_pix, image[i][j], is_H=is_H)
            # if a_crop_up.shape[0] == a_crop_down.shape[0] and  a_crop_up.shape[1] == a_crop_down.shape[1]:

            else:
                # print("is_seg==0  !!!!")  # 如果分割像素为0，那就直接在区域的1/2 + 1处分割
                if image[i][j].shape[0] == 2:
                    is_H = False
                if image[i][j].shape[1] == 2:
                    is_H = True

                if is_H == True:
                    # 分割的区域不能除以2 如果除以2的话，两个区域的shape[0]和shape[1]就一样 无法判断ind的值
                    a_crop_up, a_crop_down = crop_img((image[i][j].shape[0] // 2) + 1, image[i][j], is_H=is_H)
                else:
                    a_crop_up, a_crop_down = crop_img((image[i][j].shape[1] // 2) + 1, image[i][j], is_H=is_H)
            image.append([a_crop_up, a_crop_down])
            mask.append([np.full((a_crop_up.shape[0], a_crop_up.shape[1]), x),
                         np.full((a_crop_down.shape[0], a_crop_down.shape[1]), x + 1)])
            x += 2

    lines = lines - 1
    # for k in range(20):
    if lines != 0:
        while True:

            if mask[lines][0].shape[1] == mask[lines][1].shape[1]:
                ind = 0
            else:
                ind = 1
            mask[lines // 2 - 1][1] = np.concatenate((mask[lines][0], mask[lines][1]), axis=ind)
            lines = lines - 1

            if mask[lines][0].shape[1] == mask[lines][1].shape[1]:
                ind = 0
            else:
                ind = 1
            mask[(lines - 1) // 2][0] = np.concatenate((mask[lines][0], mask[lines][1]), axis=ind)
            lines = lines - 1
            if lines == 0: break

    if mask[0][0].shape[1] == mask[0][1].shape[1]:
        ind = 0
    else:
        ind = 1
    fin_mask = np.concatenate((mask[0][0], mask[0][1]), axis=ind)
    return fin_mask, result


if __name__ == '__main__':
    """
    lines可选参数 3、5、7、9、11、13、15、17、31、63 (奇数)
    原始 w=500, h=373
    """
    im_path = "image/fig_1.jpg"
    w = 500
    h = 373
    mask, _ = IESA(lines=3, w=w, h=h, im_path=im_path)
    image = img_as_float(io.imread(im_path))
    image = cv2.resize(image, (w, h))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    mark = mark_boundaries(image, mask, color=(1, 1, 0), outline_color=(1, 1, 0))
    ax.imshow(mark)
    plt.axis("off")
    plt.savefig("image/after_seg.jpg")
    # 显示结果
    plt.show()
    print("done")




