import cv2
import numpy as np
import os
import glob
from skimage import measure
from skimage import morphology
import math

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

def find_duandian(lung, i):
    # 此函数的功能是寻找在y=i的直线上图片有几个边缘点
    data = np.array((lung[i, 0:lung.shape[1] - 2] - lung[i, 1:lung.shape[1] - 1]))
    res1 = [np.where(data[:] != 0)][0][0]
    return res1

def find_x(image1):
    img2 = morphology.convex_hull_object(image1, connectivity=2)
    img = np.where(img2 == False, 0, 255)
    for i in range(5, img.shape[0] - 5):
        num = img.shape[0] - 6 - i
        len([np.where(img[num, :] == 255)][0][0])
        if len([np.where(img[num, :] == 255)][0][0]) >= len([np.where(img[num - 4, :] == 255)][0][0]) and len(
            [np.where(img[num, :] == 255)][0][0]) > len([np.where(img[num + 4, :] == 255)][0][0]) and len(
            find_duandian(img, i)) <= 2:
            return num, min([np.where(img[num, :] == 255)][0][0]), max([np.where(img[num, :] == 255)][0][0])
    return -1, -1, -1

def zong_seg1(img, k):
    res = []
    lung_y3, min_x1, max_x1 = find_x(img)
    for j in range(0, img.shape[0]):
        if len([np.where(img[j, :] == 255)][0][0]) > 0:
            lung_y0 = j
            break
    res.extend(np.array(np.arange(lung_y0, lung_y3, int((lung_y3 - lung_y0) / k))))
    if len(res) < k + 1:
        res.append(lung_y3)
    else:
        res[len(res) - 1] = lung_y3
    return np.array(res)

def heng_seg1(lung, lung_zong, k):
    lung_y = []
    point = []
    for i in range(0, len(lung_zong)):
        y0 = min([np.where(lung[lung_zong[i], :] == 255)][0][0])
        y1 = max([np.where(lung[lung_zong[i], :] == 255)][0][0])
        lung_y.append(np.arange(y0, y1, (y1 - y0) / k))
        point.append([y0, y1])
    lung_y = np.array(lung_y)
    x = np.array(lung_zong)
    p = []
    for j in range(1, k):
        z = np.polyfit(x, lung_y[:, j], 2)
        p.append(np.poly1d(z))
    return p, point


if __name__ == "__main__":
    images_path = glob.glob('../img_data/Screening/normal/*.*')
    mask_path = glob.glob('../img_data/Screening/normal_masks/normal_mask/*.*')

    zong_part = 3
    heng_part = 3

    if not os.path.exists('../img_data/Screening/normal_masks/partition'):
        os.makedirs('../img_data/Screening/normal_masks/partition')
    # if not os.path.exists('../img_data/Screening/normal_masks/r_mask'):
    #     os.makedirs('../img_data/Screening/normal_masks/r_mask')
    # if not os.path.exists('../img_data/Screening/normal_masks/l_mask'):
    #     os.makedirs('../img_data/Screening/normal_masks/l_mask')
    # for i in range(10):
    #     if not os.path.exists('../img_data/Screening/normal_masks/r_mask_{0}'.format(i+1)):
    #         os.makedirs('../img_data/Screening/normal_masks/r_mask_{0}'.format(i+1))
    #     if not os.path.exists('../img_data/Screening/normal_masks/l_mask_{0}'.format(i+1)):
    #         os.makedirs('../img_data/Screening/normal_masks/l_mask_{0}'.format(i+1))

    for s in range(len(images_path)):
        print("-----{0}/{1}-----".format(s + 1, len(images_path)))
        image = cv2.imread(images_path[s], cv2.IMREAD_GRAYSCALE)  # 以灰度图像格式读入图像
        image_name = os.path.basename(images_path[s]).split('.')[0]
        mask_name = ''
        mask = 0.
        for j in range(len(mask_path)):
            mask_name = os.path.basename(mask_path[j]).split('.')[0]
            if image_name in mask_name:
                mask = cv2.imread(mask_path[j], cv2.IMREAD_GRAYSCALE)  # 以灰度图像格式读入分割mask
                break

        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        right_lung, left_lung = tu_seg(mask)  # 基于肺部mask分离处左右肺
        # print(image_name)

        kernel = np.ones((5, 5), dtype=np.uint8)
        right_lung = cv2.morphologyEx(right_lung, cv2.MORPH_CLOSE, kernel)  ## 有缺陷，填补缺陷
        left_lung = cv2.morphologyEx(left_lung, cv2.MORPH_CLOSE, kernel)  ## 有缺陷，填补缺陷

        cv2.imwrite('../img_data/Screening/normal_masks/r_mask/{0}_right.png'.format(image_name), right_lung)
        cv2.imwrite('../img_data/Screening/normal_masks/l_mask/{0}_left.png'.format(image_name), left_lung)

        # 右肺分区
        rlung_zong = zong_seg1(right_lung, zong_part)  # 纵向分为上中下野，返回为肺部顶端线以及三条分界线所在行数
        rlung_p, r_point = heng_seg1(right_lung, rlung_zong[1:len(rlung_zong)], heng_part) # 横向分为内中外带，返回为中间两条划分曲线的函数
        rlung_x = np.array(range(min(rlung_zong), max(rlung_zong)))
        rlung_yvals = []
        for j in range(0, len(rlung_p)):
            rlung_yvals.append(rlung_p[j](rlung_x)) # 获取中间两条划分曲线对应的纵坐标

        # 左肺分区
        llung_zong = zong_seg1(left_lung, zong_part)
        llung_p, l_point = heng_seg1(left_lung, llung_zong[1:len(llung_zong)], heng_part)
        llung_x = np.array(range(min(llung_zong), max(llung_zong)))
        llung_yvals = []
        for j in range(0, len(llung_p)):
            llung_yvals.append(llung_p[j](llung_x))

        # # 获取各分区mask
        # # r_mask_1
        # r_mask_1 = right_lung.copy()
        # r_mask_1[rlung_zong[1]:, ::] = 0
        # for i in range(rlung_zong[0], rlung_zong[1]):
        #     r_mask_1[i, int(rlung_yvals[0][i-rlung_zong[0]]):] = 0
        # cv2.imwrite('../img_data/Screening/normal_masks/r_mask_1/{0}_r_mask1.png'.format(image_name), r_mask_1)
        # # r_mask_2
        # r_mask_2 = right_lung.copy()
        # r_mask_2[rlung_zong[1]:, ::] = 0
        # for i in range(rlung_zong[0], rlung_zong[1]):
        #     r_mask_2[i, :int(rlung_yvals[0][i - rlung_zong[0]])] = 0
        #     r_mask_2[i, int(rlung_yvals[1][i - rlung_zong[0]]):] = 0
        # cv2.imwrite('../img_data/Screening/normal_masks/r_mask_2/{0}_r_mask2.png'.format(image_name), r_mask_2)
        # # r_mask_3
        # r_mask_3 = right_lung.copy()
        # r_mask_3[rlung_zong[1]:, ::] = 0
        # for i in range(rlung_zong[0], rlung_zong[1]):
        #     r_mask_3[i, :int(rlung_yvals[1][i - rlung_zong[0]])] = 0
        # cv2.imwrite('../img_data/Screening/normal_masks/r_mask_3/{0}_r_mask3.png'.format(image_name), r_mask_3)
        # # r_mask_4
        # r_mask_4 = right_lung.copy()
        # r_mask_4[:rlung_zong[1], ::] = 0
        # r_mask_4[rlung_zong[2]:, ::] = 0
        # for i in range(rlung_zong[1], rlung_zong[2]):
        #     r_mask_4[i, int(rlung_yvals[0][i - rlung_zong[0]]):] = 0
        # cv2.imwrite('../img_data/Screening/normal_masks/r_mask_4/{0}_r_mask4.png'.format(image_name), r_mask_4)
        # # r_mask_5
        # r_mask_5 = right_lung.copy()
        # r_mask_5[:rlung_zong[1], ::] = 0
        # r_mask_5[rlung_zong[2]:, ::] = 0
        # for i in range(rlung_zong[1], rlung_zong[2]):
        #     r_mask_5[i, :int(rlung_yvals[0][i - rlung_zong[0]])] = 0
        #     r_mask_5[i, int(rlung_yvals[1][i - rlung_zong[0]]):] = 0
        # cv2.imwrite('../img_data/Screening/normal_masks/r_mask_5/{0}_r_mask5.png'.format(image_name), r_mask_5)
        # # r_mask_6
        # r_mask_6 = right_lung.copy()
        # r_mask_6[:rlung_zong[1], ::] = 0
        # r_mask_6[rlung_zong[2]:, ::] = 0
        # for i in range(rlung_zong[1], rlung_zong[2]):
        #     r_mask_6[i, :int(rlung_yvals[1][i - rlung_zong[0]])] = 0
        # cv2.imwrite('../img_data/Screening/normal_masks/r_mask_6/{0}_r_mask6.png'.format(image_name), r_mask_6)
        # # r_mask_7
        # r_mask_7 = right_lung.copy()
        # r_mask_7[:rlung_zong[2], ::] = 0
        # r_mask_7[rlung_zong[3]:, ::] = 0
        # for i in range(rlung_zong[2], rlung_zong[3]):
        #     r_mask_7[i, int(rlung_yvals[0][i - rlung_zong[0]]):] = 0
        # cv2.imwrite('../img_data/Screening/normal_masks/r_mask_7/{0}_r_mask7.png'.format(image_name), r_mask_7)
        # # r_mask_8
        # r_mask_8 = right_lung.copy()
        # r_mask_8[:rlung_zong[2], ::] = 0
        # r_mask_8[rlung_zong[3]:, ::] = 0
        # for i in range(rlung_zong[2], rlung_zong[3]):
        #     r_mask_8[i, :int(rlung_yvals[0][i - rlung_zong[0]])] = 0
        #     r_mask_8[i, int(rlung_yvals[1][i - rlung_zong[0]]):] = 0
        # cv2.imwrite('../img_data/Screening/normal_masks/r_mask_8/{0}_r_mask8.png'.format(image_name), r_mask_8)
        # # r_mask_9
        # r_mask_9 = right_lung.copy()
        # r_mask_9[:rlung_zong[2], ::] = 0
        # r_mask_9[rlung_zong[3]:, ::] = 0
        # for i in range(rlung_zong[2], rlung_zong[3]):
        #     r_mask_9[i, :int(rlung_yvals[1][i - rlung_zong[0]])] = 0
        # cv2.imwrite('../img_data/Screening/normal_masks/r_mask_9/{0}_r_mask9.png'.format(image_name), r_mask_9)
        # # r_mask_10
        # r_mask_10 = right_lung.copy()
        # r_mask_10[:rlung_zong[3], ::] = 0
        # cv2.imwrite('../img_data/Screening/normal_masks/r_mask_10/{0}_r_mask10.png'.format(image_name), r_mask_10)
        #
        # # l_mask_1
        # l_mask_1 = left_lung.copy()
        # l_mask_1[llung_zong[1]:, ::] = 0
        # for i in range(llung_zong[0], llung_zong[1]):
        #     l_mask_1[i, int(llung_yvals[0][i - llung_zong[0]]):] = 0
        # cv2.imwrite('../img_data/Screening/normal_masks/l_mask_1/{0}_l_mask1.png'.format(image_name), l_mask_1)
        # # l_mask_2
        # l_mask_2 = left_lung.copy()
        # l_mask_2[llung_zong[1]:, ::] = 0
        # for i in range(llung_zong[0], llung_zong[1]):
        #     l_mask_2[i, :int(llung_yvals[0][i - llung_zong[0]])] = 0
        #     l_mask_2[i, int(llung_yvals[1][i - llung_zong[0]]):] = 0
        # cv2.imwrite('../img_data/Screening/normal_masks/l_mask_2/{0}_l_mask2.png'.format(image_name), l_mask_2)
        # # l_mask_3
        # l_mask_3 = left_lung.copy()
        # l_mask_3[llung_zong[1]:, ::] = 0
        # for i in range(llung_zong[0], llung_zong[1]):
        #     l_mask_3[i, :int(llung_yvals[1][i - llung_zong[0]])] = 0
        # cv2.imwrite('../img_data/Screening/normal_masks/l_mask_3/{0}_l_mask3.png'.format(image_name), l_mask_3)
        # # l_mask_4
        # l_mask_4 = left_lung.copy()
        # l_mask_4[:llung_zong[1], ::] = 0
        # l_mask_4[llung_zong[2]:, ::] = 0
        # for i in range(llung_zong[1], llung_zong[2]):
        #     l_mask_4[i, int(llung_yvals[0][i - llung_zong[0]]):] = 0
        # cv2.imwrite('../img_data/Screening/normal_masks/l_mask_4/{0}_l_mask4.png'.format(image_name), l_mask_4)
        # # l_mask_5
        # l_mask_5 = left_lung.copy()
        # l_mask_5[:llung_zong[1], ::] = 0
        # l_mask_5[llung_zong[2]:, ::] = 0
        # for i in range(llung_zong[1], llung_zong[2]):
        #     l_mask_5[i, :int(llung_yvals[0][i - llung_zong[0]])] = 0
        #     l_mask_5[i, int(llung_yvals[1][i - llung_zong[0]]):] = 0
        # cv2.imwrite('../img_data/Screening/normal_masks/l_mask_5/{0}_l_mask5.png'.format(image_name), l_mask_5)
        # # l_mask_6
        # l_mask_6 = left_lung.copy()
        # l_mask_6[:llung_zong[1], ::] = 0
        # l_mask_6[llung_zong[2]:, ::] = 0
        # for i in range(llung_zong[1], llung_zong[2]):
        #     l_mask_6[i, :int(llung_yvals[1][i - llung_zong[0]])] = 0
        # cv2.imwrite('../img_data/Screening/normal_masks/l_mask_6/{0}_l_mask6.png'.format(image_name), l_mask_6)
        # # l_mask_7
        # l_mask_7 = left_lung.copy()
        # l_mask_7[:llung_zong[2], ::] = 0
        # l_mask_7[llung_zong[3]:, ::] = 0
        # for i in range(llung_zong[2], llung_zong[3]):
        #     l_mask_7[i, int(llung_yvals[0][i - llung_zong[0]]):] = 0
        # cv2.imwrite('../img_data/Screening/normal_masks/l_mask_7/{0}_l_mask7.png'.format(image_name), l_mask_7)
        # # l_mask_8
        # l_mask_8 = left_lung.copy()
        # l_mask_8[:llung_zong[2], ::] = 0
        # l_mask_8[llung_zong[3]:, ::] = 0
        # for i in range(llung_zong[2], llung_zong[3]):
        #     l_mask_8[i, :int(llung_yvals[0][i - llung_zong[0]])] = 0
        #     l_mask_8[i, int(llung_yvals[1][i - llung_zong[0]]):] = 0
        # cv2.imwrite('../img_data/Screening/normal_masks/l_mask_8/{0}_l_mask8.png'.format(image_name), l_mask_8)
        # # l_mask_9
        # l_mask_9 = left_lung.copy()
        # l_mask_9[:llung_zong[2], ::] = 0
        # l_mask_9[llung_zong[3]:, ::] = 0
        # for i in range(llung_zong[2], llung_zong[3]):
        #     l_mask_9[i, :int(llung_yvals[1][i - llung_zong[0]])] = 0
        # cv2.imwrite('../img_data/Screening/normal_masks/l_mask_9/{0}_l_mask9.png'.format(image_name), l_mask_9)
        # # l_mask_10
        # l_mask_10 = left_lung.copy()
        # l_mask_10[:llung_zong[3], ::] = 0
        # cv2.imwrite('../img_data/Screening/normal_masks/l_mask_10/{0}_l_mask10.png'.format(image_name), l_mask_10)

        #  绘制分区图像
        image[np.where(mask == 0)] = 0
        for k in range(1, len(rlung_zong)):
            cv2.line(image, (r_point[k-1][0], rlung_zong[k]), (r_point[k-1][1], rlung_zong[k]), color=(0,0,255), thickness=3)
        for m in range(len(rlung_yvals)):
            for n in range(len(rlung_yvals[m])):
                cv2.circle(image, (int(rlung_yvals[m][n]), rlung_x[n]), 0, color=(0,0,255), thickness=3)
        for k in range(1, len(llung_zong)):
            cv2.line(image, (l_point[k-1][0], llung_zong[k]), (l_point[k-1][1], llung_zong[k]), color=(0,0,255), thickness=3)
        for m in range(len(llung_yvals)):
            for n in range(len(llung_yvals[m])):
                cv2.circle(image, (int(llung_yvals[m][n]), llung_x[n]), 0, color=(0,0,255), thickness=3)
        cv2.imwrite('../img_data/Screening/normal_masks/partition/{0}_partition.png'.format(image_name), image)



