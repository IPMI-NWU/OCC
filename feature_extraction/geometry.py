import os
import cv2
import glob
import numpy as np
import pandas as pd
from collections import Counter
from skimage import measure
from skimage import morphology

def GetDiffHeight(image, mask_1, mask_2):
    """
    计算左右两肺高度差及各自高度
    :param image: 原始图像
    :param mask_1: 左/右肺mask
    :param mask_2: 右/左肺mask
    :return: 左右两肺高度差，左/右肺高度，右/左肺高度
    """
    height_1 = max([np.where(mask_1!=0)][0][0])-min([np.where(mask_1!=0)][0][0])
    height_2 = max([np.where(mask_2 != 0)][0][0]) - min([np.where(mask_2 != 0)][0][0])

    return np.abs(height_1-height_2), height_1, height_2


def find_duandian(lung, i):

    # 此函数的功能是寻找在y=i的直线上图片有几个边缘点
    data = np.array((lung[i, 0:lung.shape[1] - 2] - lung[i, 1:lung.shape[1] - 1]))
    res1 = [np.where(data[:] != 0)][0][0]
    return res1

def find_x(image1):
    """
    寻找心膈角所在行
    :param image1: 其中一侧肺的mask
    :return: 对应一侧肺部心膈角所在行，以及该行上肺的左右边缘点的列标
    """
    img = morphology.convex_hull_object(image1, connectivity=2)  # 返回逻辑二值图像
    img = np.where(img == False, 0, 255)
    for i in range(5, img.shape[0] - 5):
        num = img.shape[0] - 6 - i
        len([np.where(img[num, :] == 255)][0][0])
        if len([np.where(img[num, :] == 255)][0][0]) >= len([np.where(img[num - 2, :] == 255)][0][0]) and len(
            [np.where(img[num, :] == 255)][0][0]) > len([np.where(img[num + 2, :] == 255)][0][0]) and len(
            find_duandian(img, i)) <= 2:
            return num, min([np.where(img[num, :] == 255)][0][0]), max([np.where(img[num, :] == 255)][0][0])
    return -1, -1, -1

def GetWidth(mask, num):
    """
    将一侧肺部按num均分，返回num个宽度
    :param mask: 其中一侧肺的mask
    :param num: 需要计算宽度的间隔数
    :return: 返回num个肺部宽度
    """
    y = []
    res =[]
    y0 = min([np.where(mask!=0)][0][0])
    y1, _, _ = find_x(mask)

    y.extend(np.array(np.arange(y0, y1, int((y1- y0) / num))))
    if len(y) < num+1:
        y.append(y1)
    else:
        y[-1] = y1

    for i in range(1, len(y)):
        res.append(max([np.where(mask[y[i], :] !=0)][0][0])-min([np.where(mask[y[i], :] != 0)][0][0]))

    return np.array(res)

def MaxWidth_Thorax(r_mask, l_mask):
    """
    计算胸廓最大横径及胸部中线所在列
    :param r_mask: 右肺mask
    :param l_mask: 左肺mask
    :return: 胸廓最大横径，胸部中线所在列,胸廓最大横径所在行
    """
    y_right, _, _ = find_x(r_mask)
    y_left, _, _ = find_x(l_mask)
    y = min(y_right, y_left)  # 选取左右肺心膈角所在直线较高的行作为胸廓最大横径所在行
    x1 = min([np.where(r_mask[y, :] != 0)][0][0])
    x2 = max([np.where(l_mask[y, :] != 0)][0][0])  # x2为左肺边缘点

    return x2-x1, int(x1+(x2-x1)/2), y

def MaxWidth_Heart(r_mask, l_mask, mid):
    """
    :param r_mask: 右肺mask
    :param l_mask: 左肺mask
    :param mid: 中线
    :return: 心脏最大横径
    """
    r_bottom, _, _ = find_x(r_mask)
    l_bottom, _, _ = find_x(l_mask)

    r_top = min([np.where(r_mask!=0)][0][0])
    l_top = min([np.where(l_mask!=0)][0][0])

    r_mid = r_top + int((r_bottom-r_top)/2)
    l_mid = l_top + int((l_bottom-l_top)/2)

    min_x = mid
    for i in range(r_mid, r_bottom):
        x1 = max([np.where(r_mask[i, :]!=0)][0][0])
        if x1 <= min_x:
            min_x = x1
    max_x = mid
    for i in range(l_mid, l_bottom):
        x2 = min([np.where(l_mask[i, :] != 0)][0][0])
        if x2 >= max_x:
            max_x = x2

    return max_x-min_x, max(r_top, l_top)

def Mediastinalwidth(r_mask, l_mask, top_line, bottom_line):
    """
    :param r_mask:右肺mask
    :param l_mask: 左肺mask
    :param top_line: 左右肺顶点中较低一点所在直线
    :param bottom_line: 右肺心膈角所在直线
    :return: 上下纵隔宽度
    """
    mid_line = top_line + int((bottom_line-top_line)/2)
    MW1 = MW2 = 0
    for i in range(top_line+int((mid_line-top_line)/4), mid_line):
        x_t1 = max([np.where(r_mask[i, :]!=0)][0][0])
        x_t2 = min([np.where(l_mask[i, :]!=0)][0][0])
        if (x_t2-x_t1)>=MW1:
            MW1 = x_t2-x_t1

    for j in range(mid_line, bottom_line):
        x_b1 = max([np.where(r_mask[j, :]!=0)][0][0])
        x_b2 = min([np.where(l_mask[j, :]!=0)][0][0])
        if (x_b2-x_b1)>=MW2:
            MW2 = x_b2-x_b1

    return MW1, MW2

def CostophrenicAngle(mask):
    """
    :param mask: 其中一侧肺部mak
    :return: 对应肋膈角
    """
    bottom_x = max([np.where(mask!=0)][0][0])
    line_x=[]
    line1_y=[]
    line2_y=[]
    for i in range(bottom_x-19, bottom_x+1):
        line_x.append(i)
        line1_y.append(min([np.where(mask[i,:]!=0)][0][0]))
        line2_y.append(max([np.where(mask[i,:]!=0)][0][0]))

    poly_1 = np.polyfit(line_x, line1_y, 1)
    poly_2 = np.polyfit(line_x, line2_y, 1)

    tan_angle = np.abs(poly_1[0]-poly_2[0])/(1+poly_1[0]*poly_2[0])

    return tan_angle


if __name__ == "__main__":
    images_path = glob.glob('../img_data/Phase2_NIH/8_Normal/image/*.*')
    r_masks_path = glob.glob('../img_data/Phase2_NIH/8_Normal/r_mask/*.*')
    l_masks_path = glob.glob('../img_data/Phase2_NIH/8_Normal/l_mask/*.*')

    output = {'image_name': [], 'diff_height': [],'R_ratio_1':[], 'R_ratio_2':[], 'R_ratio_3':[], 'R_ratio_4':[], 'R_ratio_5':[],
              'L_ratio_1': [],'L_ratio_2':[], 'L_ratio_3':[], 'L_ratio_4':[], 'L_ratio_5':[], 'ctr':[], 'MW_1':[], 'MW_2':[],'R_angle':[], 'L_angle':[]}

    for i in range(0, len(r_masks_path)):
        print("-----{0}/{1}-----".format(i+1, len(r_masks_path)))
        image = cv2.imread(images_path[i], cv2.IMREAD_GRAYSCALE)
        image_name = os.path.basename(images_path[i]).split('.')[0]
        r_mask = 0
        l_mask = 0
        r_mask_name = ''
        l_mask_name = ''
        for j in range(len(r_masks_path)):
            r_mask_name = os.path.basename(r_masks_path[j]).split('.')[0]
            if image_name in r_mask_name:
                r_mask = cv2.imread(r_masks_path[j], cv2.IMREAD_GRAYSCALE)
                break

        for j in range(len(l_masks_path)):
            l_mask_name = os.path.basename(l_masks_path[j]).split('.')[0]
            if image_name in l_mask_name:
                l_mask = cv2.imread(l_masks_path[j], cv2.IMREAD_GRAYSCALE)
                break

        # image = cv2.resize(image, (r_mask.shape[1], r_mask.shape[0]))
        r_mask = cv2.resize(r_mask, (image.shape[1], image.shape[0]))
        l_mask = cv2.resize(l_mask, (image.shape[1], image.shape[0]))

        image_name = os.path.basename(images_path[i]).split('.')[0]

        output['image_name'].append(image_name)
        diff_height, r_height, l_height = GetDiffHeight(image, r_mask, l_mask)
        output['diff_height'].append(diff_height)

        r_width = GetWidth(r_mask, 5)
        l_width = GetWidth(l_mask, 5)
        output['R_ratio_1'].append(r_width[0] / r_height)
        output['R_ratio_2'].append(r_width[1] / r_height)
        output['R_ratio_3'].append(r_width[2] / r_height)
        output['R_ratio_4'].append(r_width[3] / r_height)
        output['R_ratio_5'].append(r_width[4] / r_height)
        output['L_ratio_1'].append(l_width[0] / l_height)
        output['L_ratio_2'].append(l_width[1] / l_height)
        output['L_ratio_3'].append(l_width[2] / l_height)
        output['L_ratio_4'].append(l_width[3] / l_height)
        output['L_ratio_5'].append(l_width[4] / l_height)

        throax_width, mid_line, bottom_line = MaxWidth_Thorax(r_mask, l_mask)
        heart_width, top_line = MaxWidth_Heart(r_mask, l_mask, mid_line)
        output['ctr'].append(heart_width / throax_width)  # 心胸比

        mw1, mw2 = Mediastinalwidth(r_mask, l_mask, top_line, bottom_line)
        output['MW_1'].append(mw1)
        output['MW_2'].append(mw2)

        r_angle = CostophrenicAngle(r_mask)
        l_angle = CostophrenicAngle(l_mask)
        output['R_angle'].append(r_angle)
        output['L_angle'].append(l_angle)

    df_output = pd.DataFrame(output)
    df_output.to_csv("../dataset/manual_feature/8_Normal_features/geometry_feature.csv", index=False)


