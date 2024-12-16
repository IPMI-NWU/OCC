import cv2
import numpy as np
import os
import glob
from skimage import measure
from skimage import morphology


def tu_seg(th1):
    # 此函数的功能是左右肺部分开;返回的第一个图为左肺，第二个图为右肺；
    res1 = np.zeros((th1.shape[0], th1.shape[1]))
    res2 = np.zeros((th1.shape[0], th1.shape[1]))
    label_img, num = measure.label(th1[:, :], background=0, return_num=True, connectivity=2)
    props = measure.regionprops(label_img)
    areas = np.zeros(num)
    for i in range(0, num):
        areas[i] = props[i].area
    bb = areas.tolist()
    # print('联通区域的面积分别是：', bb);
    c = []
    a = bb.copy()
    # 寻找最大联通区域的的label值；
    c.append(bb.index(max(bb)))
    # 寻找第二大联通区域的的label值；
    a.remove(max(bb))
    c.append(bb.index(max(a)))
    # print(c)
    res1[np.where(label_img == c[0] + 1)] = 255
    res2[np.where(label_img == c[1] + 1)] = 255
    for j in range(0, res1.shape[0]):
        if len([np.where(res1[j, :] == 255)][0][0]) > 0:
            y0 = j
            break
    for k in range(0, res2.shape[0]):
        if len([np.where(res2[k, :] == 255)][0][0]) > 0:
            y1 = k
            break
    # y = int(th1.shape[0] / 2)
    # y1 = int(th1.shape[0] / 3)
    # print(min([np.where(res1[y, :] == 255)][0][0]), min([np.where(res2[y, :] == 255)][0][0]))
    # if min([np.where(res1[y, :] == 255)][0][0]) > min([np.where(res2[y, :] == 255)][0][0]) or min(
    #         [np.where(res1[y1, :] == 255)][0][0]) > min([np.where(res2[y1, :] == 255)][0][0]):
    if min([np.where(res1[y0, :] == 255)][0][0]) > min([np.where(res2[y1, :] == 255)][0][0]):
        return res2, res1
    return res1, res2


if __name__ == "__main__":

    images_path = glob.glob('E:/OCC/img_data/Phase2_NIH/8_Normal/8_Normal_image/*.*')
    mask_path = glob.glob('E:/OCC/img_data/Phase2_NIH/8_Normal/8_Normal_mask/*.*')

    for i in range(len(images_path)):
        image = cv2.imread(images_path[i], cv2.IMREAD_GRAYSCALE)
        image_name = os.path.basename(images_path[i]).split('.')[0]
        mask = 0.
        for j in range(len(mask_path)):
            mask_name = os.path.basename(mask_path[j]).split('.')[0]
            if image_name in mask_name:
                mask = cv2.imread(mask_path[j], cv2.IMREAD_GRAYSCALE)  # 以灰度图像格式读入分割mask
                break
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

        right_lung, left_lung = tu_seg(mask)
        mask = right_lung + left_lung
        #
        kernel = np.ones((4, 4), dtype=np.uint8)
        closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  ## 有缺陷，填补缺陷
        cv2.imwrite('E:/OCC/img_data/Phase2_NIH/8_Normal/8_Normal_mask/{0}_mask.png'.format(image_name),closing)

        print("-----{0}/{1}-----".format(i + 1, len(images_path)))


