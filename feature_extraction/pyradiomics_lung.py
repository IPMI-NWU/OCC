import os
import cv2
import glob
import numpy as np
import pandas as pd
import radiomics
import radiomics.featureextractor as fee
import SimpleITK as sitk

if __name__ == "__main__":
    images_path = glob.glob('../img_data/Phase2_NIH/8_Normal/image/*.*')

    if not os.path.exists('../dataset/pyradiomics_feature/8_Normal_features'):
        os.makedirs('../dataset/pyradiomics_feature/8_Normal_features')

    masks_path = glob.glob('../img_data/Phase2_NIH/8_Normal/l_mask/*.*')

    # output = {'image_name':[], 'l_10Percentile':[],'l_90Percentile':[], 'l_Energy':[], 'l_Entropy':[], 'l_InterquartileRange':[],
    #               'l_Kurtosis':[], 'l_Maximum':[], 'l_MeanAbsoluteDeviation':[],'l_Mean':[], 'l_Median':[],
    #               'l_Minimum':[], 'l_Range':[], 'l_RobustMeanAbsoluteDeviation':[],'l_RootMeanSquared':[], 'l_Skewness':[],
    #               'l_TotalEnergy':[], 'l_Uniformity':[], 'l_Variance':[]}
    output = {'image_name':[], 'l_Autocorrelation':[], 'l_ClusterProminence':[], 'l_ClusterShade':[],
                  'l_ClusterTendency':[],'l_Contrast':[],'l_Correlation':[],'l_DifferenceAverage':[],
                  'l_DifferenceEntropy':[],'l_DifferenceVariance':[],'l_Id':[],'l_Idm':[],'l_Idmn':[], 'l_Idn':[],
                  'l_Imc1':[],'l_Imc2':[],'l_InverseVariance':[],'l_JointAverage':[], 'l_JointEnergy':[],
                  'l_JointEntropy':[],'l_MCC':[],'l_MaximumProbability':[],'l_SumAverage':[], 'l_SumEntropy':[],
                  'l_SumSquares':[]}
    for i in range(len(images_path)):
        print("-----{0}/{1}-----".format(i + 1, len(images_path)))
        image_name = os.path.basename(images_path[i]).split('.')[0]
        image = cv2.imread(images_path[i], cv2.IMREAD_GRAYSCALE)
        mask_path = ''
        mask = 0.
        mask_name = ''
        for j in range(len(masks_path)):
            mask_name = os.path.basename(masks_path[j]).split('.')[0]
            if image_name in mask_name:
                mask = cv2.imread(masks_path[j], cv2.IMREAD_GRAYSCALE)
                mask_path = masks_path[j]
                break
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

        image = sitk.GetImageFromArray(image)
        mask = sitk.GetImageFromArray(mask)
        extractor = fee.RadiomicsFeatureExtractor('./paras.yaml')
        # extractor.enableFeatureClassByName('firstorder')
        extractor.enableFeatureClassByName('glcm')
        # extractor.enableFeatureClassByName('glszm')
        # extractor.enableFeatureClassByName('glrlm')
        # extractor.enableImageTypeByName('LoG')
        # extractor.enableImageTypeByName('Wavelet')
        # result = extractor.execute(images_path[i], mask_path)
        result = extractor.execute(image, mask)
        #
        # for featureName in result.keys():
        #     print('Computed %s: %s' % (featureName, result[featureName]))
        # exit()
        output['image_name'].append(image_name)
        # output['l_10Percentile'].append(result['original_firstorder_10Percentile'])
        # output['l_90Percentile'].append(result['original_firstorder_90Percentile'])
        # output['l_Energy'].append(result['original_firstorder_Energy'])
        # output['l_Entropy'].append(result['original_firstorder_Entropy'])
        # output['l_InterquartileRange'].append(result['original_firstorder_InterquartileRange'])
        # output['l_Kurtosis'].append(result['original_firstorder_Kurtosis'])
        # output['l_Maximum'].append(result['original_firstorder_Maximum'])
        # output['l_MeanAbsoluteDeviation'].append(result['original_firstorder_MeanAbsoluteDeviation'])
        # output['l_Mean'].append(result['original_firstorder_Mean'])
        # output['l_Median'].append(result['original_firstorder_Median'])
        # output['l_Minimum'].append(result['original_firstorder_Minimum'])
        # output['l_Range'].append(result['original_firstorder_Range'])
        # output['l_RobustMeanAbsoluteDeviation'].append(result['original_firstorder_RobustMeanAbsoluteDeviation'])
        # output['l_RootMeanSquared'].append(result['original_firstorder_RootMeanSquared'])
        # output['l_Skewness'].append(result['original_firstorder_Skewness'])
        # output['l_TotalEnergy'].append(result['original_firstorder_TotalEnergy'])
        # output['l_Uniformity'].append(result['original_firstorder_Uniformity'])
        # output['l_Variance'].append(result['original_firstorder_Variance'])

        output['l_Autocorrelation'].append(result['original_glcm_Autocorrelation'])
        output['l_ClusterProminence'].append(result['original_glcm_ClusterProminence'])
        output['l_ClusterShade'].append(result['original_glcm_ClusterShade'])
        output['l_ClusterTendency'].append(result['original_glcm_ClusterTendency'])
        output['l_Contrast'].append(result['original_glcm_Contrast'])
        output['l_Correlation'].append(result['original_glcm_Correlation'])
        output['l_DifferenceAverage'].append(result['original_glcm_DifferenceAverage'])
        output['l_DifferenceEntropy'].append(result['original_glcm_DifferenceEntropy'])
        output['l_DifferenceVariance'].append(result['original_glcm_DifferenceVariance'])
        output['l_Id'].append(result['original_glcm_Id'])
        output['l_Idm'].append(result['original_glcm_Idm'])
        output['l_Idmn'].append(result['original_glcm_Idmn'])
        output['l_Idn'].append(result['original_glcm_Idn'])
        output['l_Imc1'].append(result['original_glcm_Imc1'])
        output['l_Imc2'].append(result['original_glcm_Imc2'])
        output['l_InverseVariance'].append(result['original_glcm_InverseVariance'])
        output['l_JointAverage'].append(result['original_glcm_JointAverage'])
        output['l_JointEnergy'].append(result['original_glcm_JointEnergy'])
        output['l_JointEntropy'].append(result['original_glcm_JointEntropy'])
        output['l_MCC'].append(result['original_glcm_MCC'])
        output['l_MaximumProbability'].append(result['original_glcm_MaximumProbability'])
        output['l_SumAverage'].append(result['original_glcm_SumAverage'])
        output['l_SumEntropy'].append(result['original_glcm_SumEntropy'])
        output['l_SumSquares'].append(result['original_glcm_SumSquares'])
        # output['l_MeshSurface'].append(result['original_shape2D_MeshSurface'])
        # output['l_PixelSurface'].append(result['original_shape2D_PixelSurface'])
        # output['l_Perimeter'].append(result['original_shape2D_Perimeter'])
        # output['l_PerimeterSurfaceRatio'].append(result['original_shape2D_PerimeterSurfaceRatio'])

        df_output = pd.DataFrame(output)
        df_output.to_csv('../dataset/pyradiomics_feature/8_Normal_features/glcm_feature_l.csv', index=False)


