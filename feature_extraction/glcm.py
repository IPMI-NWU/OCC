import os
import cv2
import glob
import numpy as np
import pandas as pd
from collections import Counter
from skimage.feature import greycomatrix, greycoprops
import matplotlib.pyplot as plt


if __name__ == "__main__":
    images_path = glob.glob('../img_data/Screening/abnormal/*.*')
    masks_path = glob.glob('../img_data/Screening/abnormal_masks/abnormal_l_mask/*.*')

    output = {'image_name': [], 'l_contrast_0': [], 'l_dissimilarity_0':[], 'l_homogeneity_0':[], 'l_energy_0':[], 'l_correlation_0':[], 'l_ASM_0':[],'l_JointAverage_0':[],'l_MaxProb_0':[],
              'l_ClusterProminence_0': [], 'l_ClusterShade_0':[], 'l_ClusterTendency_0':[], 'l_JointEnergy_0':[], 'l_JointEntropy_0':[], 'l_SumSquares_0':[], 'l_DiffAverage_0':[], 'l_DiffEntropy_0':[],
              'l_DiffVariance_0': [], 'l_IDM_0':[], 'l_IDMN_0':[], 'l_ID_0':[], 'l_IDN_0':[], 'l_InverseVariance_0':[], 'l_SumAverage_0':[], 'l_SumEntropy_0':[], 'l_IMC1_0':[], 'l_IMC2_0':[],
                               'l_contrast_45':[], 'l_dissimilarity_45':[], 'l_homogeneity_45':[], 'l_energy_45':[], 'l_correlation_45':[], 'l_ASM_45':[],'l_JointAverage_45':[],'l_MaxProb_45':[],
              'l_ClusterProminence_45': [], 'l_ClusterShade_45': [], 'l_ClusterTendency_45': [], 'l_JointEnergy_45': [], 'l_JointEntropy_45': [], 'l_SumSquares_45': [], 'l_DiffAverage_45':[], 'l_DiffEntropy_45':[],
              'l_DiffVariance_45':[], 'l_IDM_45':[], 'l_IDMN_45':[], 'l_ID_45':[], 'l_IDN_45':[], 'l_InverseVariance_45':[], 'l_SumAverage_45':[], 'l_SumEntropy_45':[], 'l_IMC1_45':[], 'l_IMC2_45':[],
                               'l_contrast_90':[], 'l_dissimilarity_90':[], 'l_homogeneity_90':[], 'l_energy_90':[], 'l_correlation_90':[], 'l_ASM_90':[],'l_JointAverage_90':[],'l_MaxProb_90':[],
              'l_ClusterProminence_90': [], 'l_ClusterShade_90': [], 'l_ClusterTendency_90': [], 'l_JointEnergy_90': [], 'l_JointEntropy_90': [], 'l_SumSquares_90': [], 'l_DiffAverage_90':[], 'l_DiffEntropy_90':[],
              'l_DiffVariance_90':[], 'l_IDM_90':[], 'l_IDMN_90':[], 'l_ID_90':[], 'l_IDN_90':[], 'l_InverseVariance_90':[], 'l_SumAverage_90':[], 'l_SumEntropy_90':[], 'l_IMC1_90':[], 'l_IMC2_90':[],
                               'l_contrast_135':[], 'l_dissimilarity_135':[], 'l_homogeneity_135':[], 'l_energy_135':[], 'l_correlation_135':[], 'l_ASM_135':[],'l_JointAverage_135':[],'l_MaxProb_135':[],
              'l_ClusterProminence_135':[], 'l_ClusterShade_135':[], 'l_ClusterTendency_135':[], 'l_JointEnergy_135':[], 'l_JointEntropy_135':[], 'l_SumSquares_135':[], 'l_DiffAverage_135':[], 'l_DiffEntropy_135':[],
              'l_DiffVariance_135':[], 'l_IDM_135':[], 'l_IDMN_135':[], 'l_ID_135':[], 'l_IDN_135':[], 'l_InverseVariance_135':[], 'l_SumAverage_135':[], 'l_SumEntropy_135':[], 'l_IMC1_135':[], 'l_IMC2_135':[],}
    for i in range(len(images_path)):
        print("-----{0}/{1}-----".format(i+1, len(images_path)))
        image = cv2.imread(images_path[i], cv2.IMREAD_GRAYSCALE)
        image_name = os.path.basename(images_path[i]).split('.')[0]
        mask = 0
        mask_name = ''
        for j in range(len(masks_path)):
            mask_name = os.path.basename(masks_path[j]).split('.')[0].split('_')[0]
            if image_name == mask_name:
                mask = cv2.imread(masks_path[j], cv2.IMREAD_GRAYSCALE)
                break

        # image = cv2.resize(image, (mask.shape[1], mask.shape[0]))
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

        image_t = min([np.where(mask != 0)][0][0])
        image_b = max([np.where(mask != 0)][0][0])
        image_l = min([np.where(mask != 0)][0][1])
        image_r = max([np.where(mask != 0)][0][1])
        image = image[image_t:image_b+1, image_l:image_r+1]

        if image_name == mask_name:

            glcm = greycomatrix(image, [1], [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4], 256, symmetric=False, normed=True)
            # print(glcm.shape)
            # plt.figure()
            # plt.subplot(2, 2, 1)
            # plt.imshow(glcm[::,::,::, 0])
            # plt.title("GLCM_0$^\circ$", fontdict={'fontsize':8})
            # plt.subplot(2, 2, 2)
            # plt.imshow(glcm[::, ::, ::, 1])
            # plt.title("GLCM_45$^\circ$", fontdict={'fontsize':8})
            # plt.subplot(2, 2, 3)
            # plt.imshow(glcm[::, ::, ::, 2])
            # plt.title("GLCM_90$^\circ$", fontdict={'fontsize':8})
            # plt.subplot(2, 2, 4)
            # plt.imshow(glcm[::, ::, ::, 3])
            # plt.title("GLCM_135$^\circ$", fontdict={'fontsize':8})
            # plt.tight_layout(rect=[0,0, 1, 0.9])
            # plt.savefig("./GLCM.png")
            # exit()

            for prop in {'contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM'}:
                temp = greycoprops(glcm, prop)
                if prop=='contrast':
                    output['l_contrast_0'].append(temp[0][0])
                    output['l_contrast_45'].append(temp[0][1])
                    output['l_contrast_90'].append(temp[0][2])
                    output['l_contrast_135'].append(temp[0][3])
                if prop=='dissimilarity':
                    output['l_dissimilarity_0'].append(temp[0][0])
                    output['l_dissimilarity_45'].append(temp[0][1])
                    output['l_dissimilarity_90'].append(temp[0][2])
                    output['l_dissimilarity_135'].append(temp[0][3])
                if prop=='homogeneity':
                    output['l_homogeneity_0'].append(temp[0][0])
                    output['l_homogeneity_45'].append(temp[0][1])
                    output['l_homogeneity_90'].append(temp[0][2])
                    output['l_homogeneity_135'].append(temp[0][3])
                if prop=='energy':
                    output['l_energy_0'].append(temp[0][0])
                    output['l_energy_45'].append(temp[0][1])
                    output['l_energy_90'].append(temp[0][2])
                    output['l_energy_135'].append(temp[0][3])
                if prop=='correlation':
                    output['l_correlation_0'].append(temp[0][0])
                    output['l_correlation_45'].append(temp[0][1])
                    output['l_correlation_90'].append(temp[0][2])
                    output['l_correlation_135'].append(temp[0][3])
                if prop=='ASM':
                    output['l_ASM_0'].append(temp[0][0])
                    output['l_ASM_45'].append(temp[0][1])
                    output['l_ASM_90'].append(temp[0][2])
                    output['l_ASM_135'].append(temp[0][3])

            glcm = glcm.transpose((3, 0, 1, 2))
            glcm_0 = glcm[0].reshape((256, 256))
            glcm_1 = glcm[1].reshape((256, 256))
            glcm_2 = glcm[2].reshape((256, 256))
            glcm_3 = glcm[3].reshape((256, 256))

            esp = 10e-6

            #  提取glcm_0特征
            max_p_0 = max([max(glcm_0[j]) for j in range(len(glcm_0))])
            output['l_MaxProb_0'].append(max_p_0)
            Px_i_0 = [0 for m in range(len(glcm_0))]
            Py_j_0 = [0 for m in range(len(glcm_0[0]))]
            Padd_k_0 = [0 for m in range(2*len(glcm_0)-1)]
            Psub_k_0 = [0 for m in range(len(glcm_0))]
            for j in range(len(glcm_0)):
                for k in range(len(glcm_0[0])):
                    Px_i_0[j] += glcm_0[j][k]
                    Py_j_0[k] += glcm_0[k][j]
            mu_x_0 = 0
            mu_y_0 = 0
            Hx_0 =0
            Hy_0 =0
            Hxy_0 = 0
            Hxy1_0 = 0
            Hxy2_0 = 0
            for n in range(len(Px_i_0)):
                mu_x_0 += (n*Px_i_0[n])
                mu_y_0 += (n*Py_j_0[n])
                Hx_0 -= Px_i_0[n]*np.log2(Px_i_0[n]+esp)
                Hy_0 -= Py_j_0[n]*np.log2(Py_j_0[n]+esp)
            for j in range(len(glcm_0)):
                for k in range(len(glcm_0[0])):
                    Hxy_0 -= glcm_0[j][k]* np.log2(glcm_0[j][k]+esp)
                    Hxy1_0 -= glcm_0[j][k] * np.log2(Px_i_0[j]*Py_j_0[k]+esp)
                    Hxy2_0 -= Px_i_0[j]*Py_j_0[k]*np.log2(Px_i_0[j]*Py_j_0[k]+esp)
            output['l_JointAverage_0'].append(mu_x_0)

            for k in range(2*len(glcm_0)-1):
                for s in range(len(glcm_0)):
                    for t in range(len(glcm_0[0])):
                        if (s+t)==k:
                            Padd_k_0[k] += glcm_0[s][t]

            for k in range(len(glcm_0)):
                for s in range(len(glcm_0)):
                    for t in range(len(glcm_0[0])):
                        if np.abs(s-t)==k:
                            Psub_k_0[k] += glcm_0[s][t]

            cluster_prominence_0 = 0
            cluster_shade_0 = 0
            cluster_tendency_0 = 0
            joint_energy_0 = 0
            joint_entropy_0 = 0
            sum_squares_0 = 0
            for s in range(len(glcm_0)):
                for t in range(len(glcm_0[0])):
                    cluster_prominence_0 += ((s+t-mu_x_0-mu_y_0)**4)*glcm_0[s][t]
                    cluster_shade_0 += ((s+t-mu_x_0-mu_y_0)**3)*glcm_0[s][t]
                    cluster_tendency_0  += ((s+t-mu_x_0-mu_y_0)**2)*glcm_0[s][t]
                    joint_energy_0 += glcm_0[s][t]**2
                    joint_entropy_0 -= glcm_0[s][t]* np.log2(glcm_0[s][t]+esp)
                    sum_squares_0 += ((s-mu_x_0)**2)*glcm_0[s][t]
            output['l_ClusterProminence_0'].append(cluster_prominence_0)
            output['l_ClusterShade_0'].append(cluster_shade_0)
            output['l_ClusterTendency_0'].append(cluster_tendency_0)
            output['l_JointEnergy_0'].append(joint_energy_0)
            output['l_JointEntropy_0'].append(joint_entropy_0)
            output['l_SumSquares_0'].append(sum_squares_0)

            diff_average_0 = 0
            diff_entropy_0 = 0
            diff_variance_0 =0
            IDM_0 = 0
            IDMN_0 =0
            ID_0 = 0
            IDN_0 = 0
            inverse_variance_0 = 0
            for k in range(len(Psub_k_0)):
                diff_average_0 += k*Psub_k_0[k]
                diff_entropy_0 += Psub_k_0[k] * np.log2(Psub_k_0[k]+esp)
                IDM_0 += Psub_k_0[k]/(1+k**2)
                IDMN_0 += Psub_k_0[k]/(1+(k/len(glcm_0))**2)
                ID_0 += Psub_k_0[k]/(1+k)
                IDN_0 += Psub_k_0[k]/(1+(k/len(glcm_0)))
                if k != 0:
                    inverse_variance_0 += Psub_k_0[k]/(k**2)
            for k in range(len(Psub_k_0)):
                diff_variance_0 += ((k-diff_average_0)**2) * Psub_k_0[k]
            output['l_DiffAverage_0'].append(diff_average_0)
            output['l_DiffEntropy_0'].append(diff_entropy_0)
            output['l_DiffVariance_0'].append(diff_average_0)
            output['l_IDM_0'].append(IDM_0)
            output['l_IDMN_0'].append(IDMN_0)
            output['l_ID_0'].append(ID_0)
            output['l_IDN_0'].append(IDN_0)
            output['l_InverseVariance_0'].append(inverse_variance_0)

            sum_average_0 = 0
            sum_entropy_0 = 0
            for s in range(len(Padd_k_0)):
                sum_average_0 += Padd_k_0[s]*(s+2)
                sum_entropy_0 -= Padd_k_0[s]*np.log2(Padd_k_0[s]+esp)
            output['l_SumAverage_0'].append(sum_average_0)
            output['l_SumEntropy_0'].append(sum_entropy_0)
            IMC1_0 = (Hxy_0-Hxy1_0)/max(Hx_0, Hy_0)
            IMC2_0 = np.sqrt(1-np.exp(-2*(Hxy2_0-Hxy_0)))
            output['l_IMC1_0'].append(IMC1_0)
            output['l_IMC2_0'].append(IMC2_0)

            ### 提取glcm_1特征

            max_p_1 = max([max(glcm_1[j]) for j in range(len(glcm_1))])
            output['l_MaxProb_45'].append(max_p_1)
            Px_i_1 = [0 for m in range(len(glcm_1))]
            Py_j_1 = [0 for m in range(len(glcm_1[0]))]
            Padd_k_1 = [0 for m in range(2 * len(glcm_1) - 1)]
            Psub_k_1 = [0 for m in range(len(glcm_1))]
            for j in range(len(glcm_1)):
                for k in range(len(glcm_1[0])):
                    Px_i_1[j] += glcm_1[j][k]
                    Py_j_1[k] += glcm_1[k][j]
            mu_x_1 = 0
            mu_y_1 = 0
            Hx_1 = 0
            Hy_1 = 0
            Hxy_1 = 0
            Hxy1_1 = 0
            Hxy2_1 = 0
            for n in range(len(Px_i_1)):
                mu_x_1 += (n * Px_i_1[n])
                mu_y_1 += (n * Py_j_1[n])
                Hx_1 -= Px_i_1[n] * np.log2(Px_i_1[n] + esp)
                Hy_1 -= Py_j_1[n] * np.log2(Py_j_1[n] + esp)
            for j in range(len(glcm_1)):
                for k in range(len(glcm_1[0])):
                    Hxy_1 -= glcm_1[j][k] * np.log2(glcm_1[j][k] + esp)
                    Hxy1_1 -= glcm_1[j][k] * np.log2(Px_i_1[j] * Py_j_1[k] + esp)
                    Hxy2_1 -= Px_i_1[j] * Py_j_1[k] * np.log2(Px_i_1[j] * Py_j_1[k] + esp)
            output['l_JointAverage_45'].append(mu_x_1)

            for k in range(2 * len(glcm_1) - 1):
                for s in range(len(glcm_1)):
                    for t in range(len(glcm_1[0])):
                        if (s + t) == k:
                            Padd_k_1[k] += glcm_1[s][t]

            for k in range(len(glcm_1)):
                for s in range(len(glcm_1)):
                    for t in range(len(glcm_1[0])):
                        if np.abs(s - t) == k:
                            Psub_k_1[k] += glcm_1[s][t]

            cluster_prominence_1 = 0
            cluster_shade_1 = 0
            cluster_tendency_1 = 0
            joint_energy_1 = 0
            joint_entropy_1 = 0
            sum_squares_1 = 0
            for s in range(len(glcm_1)):
                for t in range(len(glcm_1[0])):
                    cluster_prominence_1 += ((s + t - mu_x_1 - mu_y_1) ** 4) * glcm_1[s][t]
                    cluster_shade_1 += ((s + t - mu_x_1 - mu_y_1) ** 3) * glcm_1[s][t]
                    cluster_tendency_1 += ((s + t - mu_x_1 - mu_y_1) ** 2) * glcm_1[s][t]
                    joint_energy_1 += glcm_1[s][t] ** 2
                    joint_entropy_1 -= glcm_1[s][t] * np.log2(glcm_1[s][t] + esp)
                    sum_squares_1 += ((s - mu_x_1) ** 2) * glcm_1[s][t]
            output['l_ClusterProminence_45'].append(cluster_prominence_1)
            output['l_ClusterShade_45'].append(cluster_shade_1)
            output['l_ClusterTendency_45'].append(cluster_tendency_1)
            output['l_JointEnergy_45'].append(joint_energy_1)
            output['l_JointEntropy_45'].append(joint_entropy_1)
            output['l_SumSquares_45'].append(sum_squares_1)

            diff_average_1 = 0
            diff_entropy_1 = 0
            diff_variance_1 = 0
            IDM_1 = 0
            IDMN_1 = 0
            ID_1 = 0
            IDN_1 = 0
            inverse_variance_1 = 0
            for k in range(len(Psub_k_1)):
                diff_average_1 += k * Psub_k_1[k]
                diff_entropy_1 += Psub_k_1[k] * np.log2(Psub_k_1[k] + esp)
                IDM_1 += Psub_k_1[k] / (1 + k ** 2)
                IDMN_1 += Psub_k_1[k] / (1 + (k / len(glcm_1)) ** 2)
                ID_1 += Psub_k_1[k] / (1 + k)
                IDN_1 += Psub_k_1[k] / (1 + (k / len(glcm_1)))
                if k != 0:
                    inverse_variance_1 += Psub_k_1[k] / (k ** 2)
            for k in range(len(Psub_k_1)):
                diff_variance_1 += ((k - diff_average_1) ** 2) * Psub_k_1[k]
            output['l_DiffAverage_45'].append(diff_average_1)
            output['l_DiffEntropy_45'].append(diff_entropy_1)
            output['l_DiffVariance_45'].append(diff_average_1)
            output['l_IDM_45'].append(IDM_1)
            output['l_IDMN_45'].append(IDMN_1)
            output['l_ID_45'].append(ID_1)
            output['l_IDN_45'].append(IDN_1)
            output['l_InverseVariance_45'].append(inverse_variance_1)

            sum_average_1 = 0
            sum_entropy_1 = 0
            for s in range(len(Padd_k_1)):
                sum_average_1 += Padd_k_1[s] * (s + 2)
                sum_entropy_1 -= Padd_k_1[s] * np.log2(Padd_k_1[s] + esp)
            output['l_SumAverage_45'].append(sum_average_1)
            output['l_SumEntropy_45'].append(sum_entropy_1)
            IMC1_1 = (Hxy_1 - Hxy1_1) / max(Hx_1, Hy_1)
            IMC2_1 = np.sqrt(1 - np.exp(-2 * (Hxy2_1 - Hxy_1)))
            output['l_IMC1_45'].append(IMC1_1)
            output['l_IMC2_45'].append(IMC2_1)

            ###  提取glcm_2特征
            max_p_2 = max([max(glcm_2[j]) for j in range(len(glcm_2))])
            output['l_MaxProb_90'].append(max_p_2)
            Px_i_2 = [0 for m in range(len(glcm_2))]
            Py_j_2 = [0 for m in range(len(glcm_2[0]))]
            Padd_k_2 = [0 for m in range(2*len(glcm_2)-1)]
            Psub_k_2 = [0 for m in range(len(glcm_2))]
            for j in range(len(glcm_2)):
                for k in range(len(glcm_2[0])):
                    Px_i_2[j] += glcm_2[j][k]
                    Py_j_2[k] += glcm_2[k][j]
            mu_x_2 = 0
            mu_y_2 = 0
            Hx_2 =0
            Hy_2 =0
            Hxy_2 = 0
            Hxy1_2 = 0
            Hxy2_2 = 0
            for n in range(len(Px_i_2)):
                mu_x_2 += (n*Px_i_2[n])
                mu_y_2 += (n*Py_j_2[n])
                Hx_2 -= Px_i_2[n]*np.log2(Px_i_2[n]+esp)
                Hy_2 -= Py_j_2[n]*np.log2(Py_j_2[n]+esp)
            for j in range(len(glcm_2)):
                for k in range(len(glcm_2[0])):
                    Hxy_2 -= glcm_2[j][k]* np.log2(glcm_2[j][k]+esp)
                    Hxy1_2 -= glcm_2[j][k] * np.log2(Px_i_2[j]*Py_j_2[k]+esp)
                    Hxy2_2 -= Px_i_2[j]*Py_j_2[k]*np.log2(Px_i_2[j]*Py_j_2[k]+esp)
            output['l_JointAverage_90'].append(mu_x_2)

            for k in range(2*len(glcm_2)-1):
                for s in range(len(glcm_2)):
                    for t in range(len(glcm_2[0])):
                        if (s+t)==k:
                            Padd_k_2[k] += glcm_2[s][t]

            for k in range(len(glcm_2)):
                for s in range(len(glcm_2)):
                    for t in range(len(glcm_2[0])):
                        if np.abs(s-t)==k:
                            Psub_k_2[k] += glcm_2[s][t]

            cluster_prominence_2 = 0
            cluster_shade_2 = 0
            cluster_tendency_2 = 0
            joint_energy_2 = 0
            joint_entropy_2 = 0
            sum_squares_2 = 0
            for s in range(len(glcm_2)):
                for t in range(len(glcm_2[0])):
                    cluster_prominence_2 += ((s+t-mu_x_2-mu_y_2)**4)*glcm_2[s][t]
                    cluster_shade_2 += ((s+t-mu_x_2-mu_y_2)**3)*glcm_2[s][t]
                    cluster_tendency_2  += ((s+t-mu_x_2-mu_y_2)**2)*glcm_2[s][t]
                    joint_energy_2 += glcm_2[s][t]**2
                    joint_entropy_2 -= glcm_2[s][t]* np.log2(glcm_2[s][t]+esp)
                    sum_squares_2 += ((s-mu_x_2)**2)*glcm_2[s][t]
            output['l_ClusterProminence_90'].append(cluster_prominence_2)
            output['l_ClusterShade_90'].append(cluster_shade_2)
            output['l_ClusterTendency_90'].append(cluster_tendency_2)
            output['l_JointEnergy_90'].append(joint_energy_2)
            output['l_JointEntropy_90'].append(joint_entropy_2)
            output['l_SumSquares_90'].append(sum_squares_2)

            diff_average_2 = 0
            diff_entropy_2 = 0
            diff_variance_2 =0
            IDM_2 = 0
            IDMN_2 =0
            ID_2 = 0
            IDN_2 = 0
            inverse_variance_2 = 0
            for k in range(len(Psub_k_2)):
                diff_average_2 += k*Psub_k_2[k]
                diff_entropy_2 += Psub_k_2[k] * np.log2(Psub_k_2[k]+esp)
                IDM_2 += Psub_k_2[k]/(1+k**2)
                IDMN_2 += Psub_k_2[k]/(1+(k/len(glcm_2))**2)
                ID_2 += Psub_k_2[k]/(1+k)
                IDN_2 += Psub_k_2[k]/(1+(k/len(glcm_2)))
                if k != 0:
                    inverse_variance_2 += Psub_k_2[k]/(k**2)
            for k in range(len(Psub_k_2)):
                diff_variance_2 += ((k-diff_average_2)**2) * Psub_k_2[k]
            output['l_DiffAverage_90'].append(diff_average_2)
            output['l_DiffEntropy_90'].append(diff_entropy_2)
            output['l_DiffVariance_90'].append(diff_average_2)
            output['l_IDM_90'].append(IDM_2)
            output['l_IDMN_90'].append(IDMN_2)
            output['l_ID_90'].append(ID_2)
            output['l_IDN_90'].append(IDN_2)
            output['l_InverseVariance_90'].append(inverse_variance_2)

            sum_average_2 = 0
            sum_entropy_2 = 0
            for s in range(len(Padd_k_2)):
                sum_average_2 += Padd_k_2[s]*(s+2)
                sum_entropy_2 -= Padd_k_2[s]*np.log2(Padd_k_2[s]+esp)
            output['l_SumAverage_90'].append(sum_average_2)
            output['l_SumEntropy_90'].append(sum_entropy_2)
            IMC1_2 = (Hxy_2-Hxy1_2)/max(Hx_2, Hy_2)
            IMC2_2 = np.sqrt(1-np.exp(-2*(Hxy2_2-Hxy_2)))
            output['l_IMC1_90'].append(IMC1_2)
            output['l_IMC2_90'].append(IMC2_2)

            ###  提取glcm_3特征
            max_p_3 = max([max(glcm_3[j]) for j in range(len(glcm_3))])
            output['l_MaxProb_135'].append(max_p_3)
            Px_i_3 = [0 for m in range(len(glcm_3))]
            Py_j_3 = [0 for m in range(len(glcm_3[0]))]
            Padd_k_3 = [0 for m in range(2 * len(glcm_3) - 1)]
            Psub_k_3 = [0 for m in range(len(glcm_3))]
            for j in range(len(glcm_3)):
                for k in range(len(glcm_3[0])):
                    Px_i_3[j] += glcm_3[j][k]
                    Py_j_3[k] += glcm_3[k][j]
            mu_x_3 = 0
            mu_y_3 = 0
            Hx_3 = 0
            Hy_3 = 0
            Hxy_3 = 0
            Hxy1_3 = 0
            Hxy2_3 = 0
            for n in range(len(Px_i_3)):
                mu_x_3 += (n * Px_i_3[n])
                mu_y_3 += (n * Py_j_3[n])
                Hx_3 -= Px_i_3[n] * np.log2(Px_i_3[n] + esp)
                Hy_3 -= Py_j_3[n] * np.log2(Py_j_3[n] + esp)
            for j in range(len(glcm_3)):
                for k in range(len(glcm_3[0])):
                    Hxy_3 -= glcm_3[j][k] * np.log2(glcm_3[j][k] + esp)
                    Hxy1_3 -= glcm_3[j][k] * np.log2(Px_i_3[j] * Py_j_3[k] + esp)
                    Hxy2_3 -= Px_i_3[j] * Py_j_3[k] * np.log2(Px_i_3[j] * Py_j_3[k] + esp)
            output['l_JointAverage_135'].append(mu_x_3)

            for k in range(2 * len(glcm_3) - 1):
                for s in range(len(glcm_3)):
                    for t in range(len(glcm_3[0])):
                        if (s + t) == k:
                            Padd_k_3[k] += glcm_3[s][t]

            for k in range(len(glcm_3)):
                for s in range(len(glcm_3)):
                    for t in range(len(glcm_3[0])):
                        if np.abs(s - t) == k:
                            Psub_k_3[k] += glcm_3[s][t]

            cluster_prominence_3 = 0
            cluster_shade_3 = 0
            cluster_tendency_3 = 0
            joint_energy_3 = 0
            joint_entropy_3 = 0
            sum_squares_3 = 0
            for s in range(len(glcm_3)):
                for t in range(len(glcm_3[0])):
                    cluster_prominence_3 += ((s + t - mu_x_3 - mu_y_3) ** 4) * glcm_3[s][t]
                    cluster_shade_3 += ((s + t - mu_x_3 - mu_y_3) ** 3) * glcm_3[s][t]
                    cluster_tendency_3 += ((s + t - mu_x_3 - mu_y_3) ** 2) * glcm_3[s][t]
                    joint_energy_3 += glcm_3[s][t] ** 2
                    joint_entropy_3 -= glcm_3[s][t] * np.log2(glcm_3[s][t] + esp)
                    sum_squares_3 += ((s - mu_x_3) ** 2) * glcm_3[s][t]
            output['l_ClusterProminence_135'].append(cluster_prominence_3)
            output['l_ClusterShade_135'].append(cluster_shade_3)
            output['l_ClusterTendency_135'].append(cluster_tendency_3)
            output['l_JointEnergy_135'].append(joint_energy_3)
            output['l_JointEntropy_135'].append(joint_entropy_3)
            output['l_SumSquares_135'].append(sum_squares_3)

            diff_average_3 = 0
            diff_entropy_3 = 0
            diff_variance_3 = 0
            IDM_3 = 0
            IDMN_3 = 0
            ID_3 = 0
            IDN_3 = 0
            inverse_variance_3 = 0
            for k in range(len(Psub_k_3)):
                diff_average_3 += k * Psub_k_3[k]
                diff_entropy_3 += Psub_k_3[k] * np.log2(Psub_k_3[k] + esp)
                IDM_3 += Psub_k_3[k] / (1 + k ** 2)
                IDMN_3 += Psub_k_3[k] / (1 + (k / len(glcm_3)) ** 2)
                ID_3 += Psub_k_3[k] / (1 + k)
                IDN_3 += Psub_k_3[k] / (1 + (k / len(glcm_3)))
                if k != 0:
                    inverse_variance_3 += Psub_k_3[k] / (k ** 2)
            for k in range(len(Psub_k_3)):
                diff_variance_3 += ((k - diff_average_3) ** 2) * Psub_k_3[k]
            output['l_DiffAverage_135'].append(diff_average_3)
            output['l_DiffEntropy_135'].append(diff_entropy_3)
            output['l_DiffVariance_135'].append(diff_average_3)
            output['l_IDM_135'].append(IDM_3)
            output['l_IDMN_135'].append(IDMN_3)
            output['l_ID_135'].append(ID_3)
            output['l_IDN_135'].append(IDN_3)
            output['l_InverseVariance_135'].append(inverse_variance_3)

            sum_average_3 = 0
            sum_entropy_3 = 0
            for s in range(len(Padd_k_3)):
                sum_average_3 += Padd_k_3[s] * (s + 2)
                sum_entropy_3 -= Padd_k_3[s] * np.log2(Padd_k_3[s] + esp)
            output['l_SumAverage_135'].append(sum_average_3)
            output['l_SumEntropy_135'].append(sum_entropy_3)
            IMC1_3 = (Hxy_3 - Hxy1_3) / max(Hx_3, Hy_3)
            IMC2_3 = np.sqrt(1 - np.exp(-2 * (Hxy2_3 - Hxy_3)))
            output['l_IMC1_135'].append(IMC1_3)
            output['l_IMC2_135'].append(IMC2_3)

            output['image_name'].append(image_name)

    df_output = pd.DataFrame(output)
    df_output.to_csv('../dataset/manual_feature/abnormal_features_3/glcm_feature_l.csv', index=False)

