import os
import cv2
import glob
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

if __name__ == "__main__":
    images_path = glob.glob('../img_data/Phase2_NIH/4_Mass/image/*.*')
    if not os.path.exists('../dataset/manual_feature/4_Mass_features'):
        os.makedirs('../dataset/manual_feature/4_Mass_features')

    for k in range(1,11):
        masks_path = glob.glob('../img_data/Phase2_NIH/4_Mass/l_mask_{0}/*.*'.format(k))
        output = {'image_name': [], 'l_{0}_mean'.format(k): [], 'l_{0}_std'.format(k): [],
                  'l_{0}_min'.format(k): [], 'l_{0}_max'.format(k): [], 'l_{0}_median'.format(k): [],
                  'l_{0}_iqr'.format(k): [], 'l_{0}_skew'.format(k): [], 'l_{0}_kurt'.format(k): [],
                  'l_{0}_energy'.format(k): [], 'l_{0}_entropy'.format(k): [],
                  'l_{0}_MAD'.format(k): [], 'l_{0}_RMS'.format(k): [], 'l_{0}_uniformity'.format(k): []}
        # 均值， 方差， 最小值， 最大值， 中值， 四分位范围， 偏度， 峰度

        for i in range(len(images_path)):
            print("-----{0}/{1}-----".format(i + 1, len(images_path)))
            image = cv2.imread(images_path[i], cv2.IMREAD_GRAYSCALE)
            image_name = os.path.basename(images_path[i]).split('.')[0]
            mask = 0
            mask_name = ''
            for m in range(len(masks_path)):
                mask_name = os.path.basename(masks_path[m]).split('.')[0]
                if image_name in mask_name:
                    mask = cv2.imread(masks_path[m], cv2.IMREAD_GRAYSCALE)
                    break
            # image = cv2.resize(image, (mask.shape[1], mask.shape[0]))
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

            roi_image = image[np.where(mask != 0)]
            df_roi = pd.DataFrame()
            df_roi['roi'] = roi_image
            mean = df_roi.describe().values[1][0]  # 均值

            pixel_num = len(roi_image)  # roi像素点个数
            count = Counter(roi_image)
            count_value = np.array(list(count.values()))  # 每个像素值的个数
            energy = 0  # 能量
            entropy = 0  # 熵
            MAD = 0  # 平均绝对差
            uniformity = 0  # 均匀性
            for j in range(len(roi_image)):
                energy += pow(roi_image[j], 2) / len(images_path)  # 除以len(images_path)防溢出
                MAD += (roi_image[j] - mean)
            MAD = MAD / pixel_num
            RMS = np.sqrt(energy / pixel_num)  # 均方根
            for t in range(len(count_value)):
                entropy -= ((count_value[t] / pixel_num) * np.log2(count_value[t] / pixel_num))
                uniformity += (count_value[t] / pixel_num) ** 2

            # print(df_roi.describe())   # count,mean,std,min,25%,50%,75%,max,
            output['image_name'].append(image_name)
            output['l_{0}_mean'.format(k)].append(df_roi.describe().values[1][0])
            output['l_{0}_std'.format(k)].append(df_roi.describe().values[2][0])
            output['l_{0}_min'.format(k)].append(df_roi.describe().values[3][0])
            output['l_{0}_max'.format(k)].append(df_roi.describe().values[7][0])
            output['l_{0}_median'.format(k)].append(df_roi.describe().values[5][0])
            output['l_{0}_iqr'.format(k)].append(df_roi.describe().values[6][0] - df_roi.describe().values[4][0])
            output['l_{0}_skew'.format(k)].append(df_roi.skew().values[0])
            output['l_{0}_kurt'.format(k)].append(df_roi.kurt().values[0])
            output['l_{0}_energy'.format(k)].append(energy)
            output['l_{0}_entropy'.format(k)].append(entropy)
            output['l_{0}_MAD'.format(k)].append(MAD)
            output['l_{0}_RMS'.format(k)].append(RMS)
            output['l_{0}_uniformity'.format(k)].append(uniformity)
        df_output = pd.DataFrame(output)
        df_output.to_csv('../dataset/manual_feature/4_Mass_features/fosf_feature_l{0}.csv'.format(k),
                         index=False)




