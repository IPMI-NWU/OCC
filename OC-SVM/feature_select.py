import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_auc_score, roc_curve, mutual_info_score
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler
from scipy.stats import ttest_ind, levene
import seaborn as sns
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
import matplotlib.pyplot as plt

def select_feature_by_Spec(df_normal_set, df_abnormal_set):
    normal_set = df_normal_set.values
    abnormal_set = df_abnormal_set.values
    normal_label = np.array([0 for i in range(len(normal_set))]).reshape(-1, 1)  # 测试集正常label
    abnormal_label = np.array([1 for i in range(len(abnormal_set))]).reshape(-1, 1)  # 测试集异常label

    valid_feature = np.concatenate((normal_set, abnormal_set), axis=0)  # 整合后测试集特征
    valid_label = np.array(np.concatenate((normal_label, abnormal_label), axis=0), dtype=np.int64)  # 测试集label

    scaler = StandardScaler()

    valid_feature = scaler.fit_transform(valid_feature)
    columns = normal_set.shape[1]
    select_features = []

    for i in range(columns):
        sensitivity = 0.
        specificity = 0.
        th = 0.0
        while sensitivity < 1:
            th += 0.05
            if th >= 1:
                break
            temp_sensitivity, temp_specificity = getClassfyRes(valid_feature[::, i], valid_label, th)
            if sensitivity < temp_sensitivity:
                sensitivity = temp_sensitivity
                specificity = temp_specificity
        if sensitivity >= 0.6:
            select_features.append(df_normal_set.columns[i])
    return select_features

def select_feature_by_Auc(df_normal_set, df_abnormal_set):
    normal_set = df_normal_set.values
    abnormal_set = df_abnormal_set.values
    normal_label = np.array([0 for i in range(len(normal_set))]).reshape(-1, 1)  # 测试集正常label
    abnormal_label = np.array([1 for i in range(len(abnormal_set))]).reshape(-1, 1)  # 测试集异常label

    valid_feature = np.concatenate((normal_set, abnormal_set), axis=0)  # 整合后测试集特征
    valid_label = np.array(np.concatenate((normal_label, abnormal_label), axis=0), dtype=np.int64)  # 测试集label

    scaler = StandardScaler()

    valid_feature = scaler.fit_transform(valid_feature)
    columns = normal_set.shape[1]
    select_features = []

    for i in range(columns):
        fpr, tpr, _ = roc_curve(valid_label, valid_feature[::, i])
        auc_score = auc(fpr, tpr)
        if auc_score >= 0.68 or auc_score <= 0.13:
            select_features.append(df_normal_set.columns[i])

    return select_features

def getClassfyRes(feature, lable, th):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(feature)):
        if feature[i] >= th and lable[i] == 1:
            TP += 1
        elif feature[i] >= th and lable[i] == 0:
            FP += 1
        elif feature[i] < th and lable[i] == 1:
            FN += 1
        else:
            TN += 1
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    return sensitivity, specificity

def select_feature_by_Ttest(df_normal_set, df_abnormal_set):
    index = []
    for colName in df_normal_set.columns[:]:
        if levene(df_normal_set[colName], df_abnormal_set[colName])[1] > 0.05:  # 有方差齐性
            if ttest_ind(df_normal_set[colName], df_abnormal_set[colName])[1] < 0.05:
                index.append(colName)
        else:  # 不具方差齐性
            if ttest_ind(df_normal_set[colName], df_abnormal_set[colName], equal_var=False)[1] < 0.05:
                index.append(colName)
    return index

def getSet(dataset, path):
    select_image = []
    file = open(path, encoding='utf-8')
    for line in file:
        select_image.append(line.strip('\n'))
    index_list = []
    for idx, row in dataset.iterrows():
        if row['image_name'] not in select_image:
            index_list.append(idx)

    return dataset.drop(index_list)

def factorAnalyze(df_normal_set, df_abnormal_set, factor_num):
    if df_abnormal_set is not None:
        df_dataset = pd.concat([df_normal_set, df_abnormal_set])
    else:
        df_dataset = df_normal_set
    # 充分性检验，检验数据集中是否能找到这些factor
    # print(calculate_bartlett_sphericity(df_dataset)[1])
    # print(calculate_kmo(df_dataset)[1])

    # # 对现有所有特征进行因子分析（因子个数=特征数），排序判断所需选择的因子个数
    # fa = FactorAnalyzer(df_dataset.shape[1], rotation=None)
    # fa.fit(df_dataset)
    # ev, v = fa.get_eigenvalues()
    # plt.scatter(range(1, df_dataset.shape[1] + 1), ev)
    # plt.plot(range(1, df_dataset.shape[1] + 1), ev)
    # plt.title('Scree Plot')
    # plt.xlabel('Factors')
    # plt.ylabel('Eigenvalue')
    # plt.grid()
    # plt.show()


    # factor_num 因子个数
    fa = FactorAnalyzer(factor_num, rotation=None)
    fa.fit(df_dataset)
    # print(fa.loadings_.shape)
    return pd.DataFrame(fa.transform(df_dataset)), fa