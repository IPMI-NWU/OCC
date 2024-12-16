import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler
import joblib
from functools import reduce
import heapq
import matplotlib.pyplot as plt
import seaborn as sns
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
from scipy.stats import ttest_ind, levene
from feature_select import select_feature_by_Spec, select_feature_by_Auc, select_feature_by_Ttest, factorAnalyze
from sklearn import metrics

def getImageNameList(path):
    select_image = []
    file = open(path, encoding='utf-8')
    for line in file:
        select_image.append(line.strip('\n'))
    return select_image


def getSelectSet(dataset, select_list):
    index_list = []
    for idx, row in dataset.iterrows():
        if row['image_name'] not in select_list:
            index_list.append(idx)

    return dataset.drop(index_list)


def getSet(dataset, path):
    select_list = getImageNameList(path)
    return getSelectSet(dataset, select_list)


def read_features_name(path):
    features_list = []
    file_features = open(path, encoding='utf-8')
    for line in file_features:
        features_list.append(line.strip('\n'))
    return features_list


def get_all_features(type):
    if type == "geo":
        df_normal_set = pd.read_csv("../dataset/manual_feature/normal_features_3/geometry_feature.csv", header=0)
        df_abnormal_set = pd.read_csv("../dataset/manual_feature/abnormal_features_3/geometry_feature.csv", header=0)
    elif type == "left_fosf":
        df_normal_list = []
        df_abnormal_list = []
        for i in range(10):
            df_normal_list.append(
                pd.read_csv("../dataset/manual_feature/normal_features_3/fosf_feature_l{0}.csv".format(i + 1),
                            header=0))
            df_abnormal_list.append(
                pd.read_csv("../dataset/manual_feature/abnormal_features_3/fosf_feature_l{0}.csv".format(i + 1),
                            header=0))
        df_normal_list.append(pd.read_csv("../dataset/manual_feature/normal_features_3/fosf_feature_l.csv", header=0))
        df_abnormal_list.append(
            pd.read_csv("../dataset/manual_feature/abnormal_features_3/fosf_feature_l.csv", header=0))
        df_normal_set = reduce(lambda left, right: pd.merge(left, right, on=['image_name']), df_normal_list)
        df_abnormal_set = reduce(lambda left, right: pd.merge(left, right, on=['image_name']), df_abnormal_list)
    elif type == "right_fosf":
        df_normal_list = []
        df_abnormal_list = []
        for i in range(10):
            df_normal_list.append(
                pd.read_csv("../dataset/manual_feature/normal_features_3/fosf_feature_r{0}.csv".format(i + 1),
                            header=0))
            df_abnormal_list.append(
                pd.read_csv("../dataset/manual_feature/abnormal_features_3/fosf_feature_r{0}.csv".format(i + 1),
                            header=0))
        df_normal_list.append(pd.read_csv("../dataset/manual_feature/normal_features_3/fosf_feature_r.csv", header=0))
        df_abnormal_list.append(
            pd.read_csv("../dataset/manual_feature/abnormal_features_3/fosf_feature_r.csv", header=0))
        df_normal_set = reduce(lambda left, right: pd.merge(left, right, on=['image_name']), df_normal_list)
        df_abnormal_set = reduce(lambda left, right: pd.merge(left, right, on=['image_name']), df_abnormal_list)
    elif type == "left_glcm":
        df_normal_set = pd.read_csv("../dataset/pyradiomics_feature/normal_features_3/glcm_feature_l.csv", header=0)
        df_abnormal_set = pd.read_csv("../dataset/pyradiomics_feature/abnormal_features_3/glcm_feature_l.csv", header=0)
    else:
        df_normal_set = pd.read_csv("../dataset/pyradiomics_feature/normal_features_3/glcm_feature_r.csv", header=0)
        df_abnormal_set = pd.read_csv("../dataset/pyradiomics_feature/abnormal_features_3/glcm_feature_r.csv", header=0)

    return df_normal_set, df_abnormal_set


def get_split_Set(df_normal_set, df_abnormal_set, type="all", train_root=None, valid_normal_root=None,
                  valid_abnormal_root=None, test_normal_root=None, test_abnormal_root=None):
    if valid_normal_root is not None:
        df_valid_normal = getSet(df_normal_set, valid_normal_root)
    if valid_abnormal_root is not None:
        df_valid_abnormal = getSet(df_abnormal_set, valid_abnormal_root)
    if train_root is not None:
        df_train_set = getSet(df_normal_set, train_root)
    if test_normal_root is not None:
        df_test_normal = getSet(df_normal_set, test_normal_root)
    if test_abnormal_root is not None:
        df_test_abnormal = getSet(df_abnormal_set, test_abnormal_root)

    if type == "train":
        return df_train_set, df_valid_normal, df_valid_abnormal
    elif type == "test":
        return df_test_normal, df_test_abnormal
    else:
        return df_train_set, df_valid_normal, df_valid_abnormal, df_test_normal, df_test_abnormal


def select_feature_train(df_train_set, df_valid_normal, df_valid_abnormal, type):
    df_valid_normal = df_valid_normal.drop(columns='image_name')
    df_valid_abnormal = df_valid_abnormal.drop(columns='image_name')

    train_image_name = df_train_set['image_name'].values

    if type == "geo":
        # # #基于T检验的选择
        features_list = select_feature_by_Ttest(df_valid_normal, df_valid_abnormal)
        np.savetxt('./result/geometry_classify/Ttest_features.txt', features_list, fmt='%s')
        ####多变量分析
        # 基于因子分析
        train_set, geo_factor = factorAnalyze(df_train_set.drop(columns='image_name'), None, 5)
        train_set = train_set.values
        # 因子分析+单变量
        train_set_singleSelect = df_train_set[features_list].values
        train_set = np.concatenate((train_set, train_set_singleSelect), axis=1)
        joblib.dump(geo_factor, "./result/models/factors/geo_factor")


    elif type == "left_fosf":
        # # #基于Spec的选择
        features_list = select_feature_by_Spec(df_valid_normal, df_valid_abnormal)
        np.savetxt('./result/leftLung_Fosf_classify/spec_features.txt', features_list, fmt='%s')
        ####多变量分析
        # 基于因子分析
        train_set, leftFosf_factor = factorAnalyze(df_train_set.drop(columns='image_name'), None, 15)
        train_set = train_set.values
        # 因子分析+单变量
        train_set_singleSelect = df_train_set[features_list].values
        train_set = np.concatenate((train_set, train_set_singleSelect), axis=1)
        joblib.dump(leftFosf_factor, "./result/models/factors/leftFosf_factor")

    elif type == "right_fosf":
        # # #基于Ttest的选择
        features_list = select_feature_by_Ttest(df_valid_normal, df_valid_abnormal)
        np.savetxt('./result/rightLung_Fosf_classify/Ttest_features.txt', features_list, fmt='%s')
        df_train_set = df_train_set[features_list]
        train_set = df_train_set.values

    elif type == "left_glcm":
        # # #基于Spec的选择
        features_list = select_feature_by_Spec(df_valid_normal, df_valid_abnormal)
        np.savetxt('./result/leftLung_Glcm_classify/spec_features.txt', features_list, fmt='%s')
        df_train_set = df_train_set[features_list]
        train_set = df_train_set.values
    else:
        # # #基于Spec的选择
        features_list = select_feature_by_Spec(df_valid_normal, df_valid_abnormal)
        np.savetxt('./result/rightLung_Glcm_classify/spec_features.txt', features_list, fmt='%s')
        df_train_set = df_train_set[features_list]
        train_set = df_train_set.values

    return train_image_name, train_set


def select_feature_test(df_test_normal, df_test_abnormal, type):
    test_image_name = pd.concat((df_test_normal['image_name'], df_test_abnormal['image_name']), axis=0).values

    test_normal_label = [0 for i in range(df_test_normal.shape[0])]  # 测试集正常label
    test_abnormal_label = [1 for i in range(df_test_abnormal.shape[0])]  # 测试集异常label
    test_label = np.array(np.concatenate((test_normal_label, test_abnormal_label), axis=0), dtype=np.int64)  # 测试集label

    if type == "geo":
        features_list = read_features_name('./result/geometry_classify/Ttest_features.txt')
        # 基于因子分析
        geo_factor = joblib.load("./result/models/factors/geo_factor")
        df_test_feature = pd.concat(
            (df_test_normal.drop(columns='image_name'), df_test_abnormal.drop(columns='image_name')), axis=0)
        test_feature = pd.DataFrame(geo_factor.transform(df_test_feature)).values

        # 因子分析+单变量
        test_normal_singleSelect = df_test_normal[features_list].values
        test_abnormal_singleSelect = df_test_abnormal[features_list].values
        test_singleSelect = np.concatenate((test_normal_singleSelect, test_abnormal_singleSelect), axis=0)
        test_feature = np.concatenate((test_feature, test_singleSelect), axis=1)

    elif type == "left_fosf":
        # # #基于Spec的选择
        features_list = read_features_name('./result/leftLung_Fosf_classify/spec_features.txt')
        # 基于因子分析
        leftFosf_factor = joblib.load("./result/models/factors/leftFosf_factor")
        df_test_feature = pd.concat(
            (df_test_normal.drop(columns='image_name'), df_test_abnormal.drop(columns='image_name')), axis=0)
        test_feature = pd.DataFrame(leftFosf_factor.transform(df_test_feature)).values
        # 因子分析+单变量
        test_normal_singleSelect = df_test_normal[features_list].values
        test_abnormal_singleSelect = df_test_abnormal[features_list].values
        test_singleSelect = np.concatenate((test_normal_singleSelect, test_abnormal_singleSelect), axis=0)
        test_feature = np.concatenate((test_feature, test_singleSelect), axis=1)

    elif type == "right_fosf":
        # # #基于Ttest的选择
        features_list = read_features_name('./result/rightLung_Fosf_classify/Ttest_features.txt')
        df_test_normal = df_test_normal[features_list]
        df_test_abnormal = df_test_abnormal[features_list]
        test_feature = pd.concat((df_test_normal, df_test_abnormal), axis=0).values  # 整合后测试集特征

    elif type == "left_glcm":
        # # #基于Spec的选择
        features_list = read_features_name('./result/leftLung_Glcm_classify/spec_features.txt')
        df_test_normal = df_test_normal[features_list]
        df_test_abnormal = df_test_abnormal[features_list]
        test_feature = pd.concat((df_test_normal, df_test_abnormal), axis=0).values  # 整合后测试集特征
    else:
        # # #基于Spec的选择
        features_list = read_features_name('./result/rightLung_Glcm_classify/spec_features.txt')
        df_test_normal = df_test_normal[features_list]
        df_test_abnormal = df_test_abnormal[features_list]
        test_feature = pd.concat((df_test_normal, df_test_abnormal), axis=0).values  # 整合后测试集特征

    return test_image_name, test_feature, test_label


def clf_train(train_set, train_image_name, nu, type):
    if type == "geo":
        model_path = './result/models/oc-svm_geometry_{0}.pkl'.format(nu)
        scaler_path = "./result/models/standardScaler/geo_scaler"
        train_result = {'image_name': [], 'geo_pre_label': [], 'geo_anomaly_score': []}
    elif type == "left_fosf":
        model_path = './result/models/oc-svm_LeftFosf_{0}.pkl'.format(nu)
        scaler_path = "./result/models/standardScaler/leftFosf_scaler"
        train_result = {'image_name': [], 'left_fosf_pre_label': [], 'left_fosf_anomaly_score': []}
    elif type == "right_fosf":
        model_path = './result/models/oc-svm_RightFosf_{0}.pkl'.format(nu)
        scaler_path = "./result/models/standardScaler/rightFosf_scaler"
        train_result = {'image_name': [], 'right_fosf_pre_label': [], 'right_fosf_anomaly_score': []}
    elif type == "left_glcm":
        model_path = './result/models/oc-svm_LeftGlcm_{0}.pkl'.format(nu)
        scaler_path = "./result/models/standardScaler/leftGlcm_scaler"
        train_result = {'image_name': [], 'left_glcm_pre_label': [], 'left_glcm_anomaly_score': []}
    else:
        model_path = './result/models/oc-svm_RightGlcm_{0}.pkl'.format(nu)
        scaler_path = "./result/models/standardScaler/rightGlcm_scaler"
        train_result = {'image_name': [], 'right_glcm_pre_label': [], 'right_glcm_anomaly_score': []}

    scaler = StandardScaler()
    scaler.fit(train_set)
    train_set = scaler.transform(train_set)
    joblib.dump(scaler, scaler_path)

    clf = OneClassSVM(kernel='rbf', degree=3, gamma='scale', nu=nu)
    clf.fit(train_set)
    joblib.dump(clf, model_path)

    model = joblib.load(model_path)
    pre_label_train = model.predict(train_set)
    pre_label_train[np.where(pre_label_train == 1)] = 0
    pre_label_train[np.where(pre_label_train == -1)] = 1
    anomaly_score_train = model.decision_function(train_set)
    train_result['image_name'].extend(train_image_name)
    if type == "geo":
        train_result['geo_pre_label'].extend(pre_label_train)
        train_result['geo_anomaly_score'].extend(anomaly_score_train)
    elif type == "left_fosf":
        train_result['left_fosf_pre_label'].extend(pre_label_train)
        train_result['left_fosf_anomaly_score'].extend(anomaly_score_train)
    elif type == "right_fosf":
        train_result['right_fosf_pre_label'].extend(pre_label_train)
        train_result['right_fosf_anomaly_score'].extend(anomaly_score_train)
    elif type == "left_glcm":
        train_result['left_glcm_pre_label'].extend(pre_label_train)
        train_result['left_glcm_anomaly_score'].extend(anomaly_score_train)
    else:
        train_result['right_glcm_pre_label'].extend(pre_label_train)
        train_result['right_glcm_anomaly_score'].extend(anomaly_score_train)

    train_result = pd.DataFrame(train_result)

    return train_result, model_path, scaler_path


def clf_test(test_feature, test_image_name, model_path, scaler_path, type):
    scaler = joblib.load(scaler_path)
    test_feature = scaler.transform(test_feature)

    model = joblib.load(model_path)
    pre_label = model.predict(test_feature)
    pre_label[np.where(pre_label == 1)] = 0
    pre_label[np.where(pre_label == -1)] = 1
    anomaly_score = model.decision_function(test_feature)

    if type == "geo":
        test_result = {'image_name': [], 'geo_pre_label': [], 'geo_anomaly_score': []}
        test_result['geo_pre_label'].extend(pre_label)
        test_result['geo_anomaly_score'].extend(anomaly_score)
    elif type == "left_fosf":
        test_result = {'image_name': [], 'left_fosf_pre_label': [], 'left_fosf_anomaly_score': []}
        test_result['left_fosf_pre_label'].extend(pre_label)
        test_result['left_fosf_anomaly_score'].extend(anomaly_score)
    elif type == "right_fosf":
        test_result = {'image_name': [], 'right_fosf_pre_label': [], 'right_fosf_anomaly_score': []}
        test_result['right_fosf_pre_label'].extend(pre_label)
        test_result['right_fosf_anomaly_score'].extend(anomaly_score)
    elif type == "left_glcm":
        test_result = {'image_name': [], 'left_glcm_pre_label': [], 'left_glcm_anomaly_score': []}
        test_result['left_glcm_pre_label'].extend(pre_label)
        test_result['left_glcm_anomaly_score'].extend(anomaly_score)
    else:
        test_result = {'image_name': [], 'right_glcm_pre_label': [], 'right_glcm_anomaly_score': []}
        test_result['right_glcm_pre_label'].extend(pre_label)
        test_result['right_glcm_anomaly_score'].extend(anomaly_score)

    test_result['image_name'].extend(test_image_name)
    test_result = pd.DataFrame(test_result)

    return test_result


if __name__ == "__main__":
    train_root = "../dataset/split_dataset/train_set.txt"
    valid_normal_root = "../dataset/split_dataset/valid_normal.txt"
    valid_abnormal_root = "../dataset/split_dataset/valid_abnormal.txt"
    test_normal_root = "../dataset/split_dataset/test_normal.txt"

    df_geo_normal, df_geo_abnormal = get_all_features("geo")
    df_leftFosf_normal, df_leftFosf_abnormal = get_all_features("left_fosf")
    df_rightFosf_normal, df_rightFosf_abnormal = get_all_features("right_fosf")
    df_leftGlcm_normal, df_leftGlcm_abnormal = get_all_features("left_glcm")
    df_rightGlcm_normal, df_rightGlcm_abnormal = get_all_features("right_glcm")

    geo_model_path = "./result/models/oc-svm_geometry_0.25.pkl"
    leftFosf_model_path = "./result/models/oc-svm_LeftFosf_0.08.pkl"
    rightFosf_model_path = "./result/models/oc-svm_RightFosf_0.1.pkl"
    leftGlcm_model_path = "./result/models/oc-svm_LeftGlcm_0.15.pkl"
    rightGlcm_model_path = "./result/models/oc-svm_RightGlcm_0.2.pkl"


    geo_scaler_path = "./result/models/standardScaler/geo_scaler"
    leftFosf_scaler_path = "./result/models/standardScaler/leftFosf_scaler"
    rightFosf_scaler_path = "./result/models/standardScaler/rightFosf_scaler"
    leftGlcm_scaler_path = "./result/models/standardScaler/leftGlcm_scaler"
    rightGlcm_scaler_path = "./result/models/standardScaler/rightGlcm_scaler"

    ensemble_model_path = "./result/Stacking/ensemble_all/ensemble_model.pkl"

    error_list = []
    test_abnormal_root = "../dataset/split_dataset/test_abnormal.txt"
    df_geo_test_normal, df_geo_test_abnormal = get_split_Set(df_geo_normal, df_geo_abnormal, type="test",
                                                             test_normal_root=test_normal_root,
                                                             test_abnormal_root=test_abnormal_root)
    df_leftFosf_test_normal, df_leftFosf_test_abnormal = get_split_Set(df_leftFosf_normal, df_leftFosf_abnormal,
                                                                       type="test",
                                                                       test_normal_root=test_normal_root,
                                                                       test_abnormal_root=test_abnormal_root)
    df_rightFosf_test_normal, df_rightFosf_test_abnormal = get_split_Set(df_rightFosf_normal, df_rightFosf_abnormal,
                                                                         type="test",
                                                                         test_normal_root=test_normal_root,
                                                                         test_abnormal_root=test_abnormal_root)
    df_leftGlcm_test_normal, df_leftGlcm_test_abnormal = get_split_Set(df_leftGlcm_normal, df_leftGlcm_abnormal,
                                                                       type="test",
                                                                       test_normal_root=test_normal_root,
                                                                       test_abnormal_root=test_abnormal_root)
    df_rightGlcm_test_normal, df_rightGlcm_test_abnormal = get_split_Set(df_rightGlcm_normal, df_rightGlcm_abnormal,
                                                                         type="test",
                                                                         test_normal_root=test_normal_root,
                                                                         test_abnormal_root=test_abnormal_root)
    geo_test_image_name, geo_test_feature, geo_test_label = select_feature_test(df_geo_test_normal,
                                                                                df_geo_test_abnormal, "geo")

    leftFosf_test_image_name, leftFosf_test_feature, leftFosf_test_label = select_feature_test(
        df_leftFosf_test_normal, df_leftFosf_test_abnormal, "left_fosf")
    rightFosf_test_image_name, rightFosf_test_feature, rightFosf_test_label = select_feature_test(
        df_rightFosf_test_normal, df_rightFosf_test_abnormal, "right_fosf")
    leftGlcm_test_image_name, leftGlcm_test_feature, leftGlcm_test_label = select_feature_test(
        df_leftGlcm_test_normal, df_leftGlcm_test_abnormal, "left_glcm")
    rightGlcm_test_image_name, rightGlcm_test_feature, rightGlcm_test_label = select_feature_test(
        df_rightGlcm_test_normal, df_rightGlcm_test_abnormal, "right_glcm")

    df_test_result_list = []
    geo_test_res = clf_test(geo_test_feature, geo_test_image_name, geo_model_path, geo_scaler_path,
                            "geo")
    leftFosf_test_res = clf_test(leftFosf_test_feature, leftFosf_test_image_name,
                                 leftFosf_model_path,
                                 leftFosf_scaler_path, "left_fosf")
    rightFosf_test_res = clf_test(rightFosf_test_feature, rightFosf_test_image_name,
                                  rightFosf_model_path, rightFosf_scaler_path, "right_fosf")
    leftGlcm_test_res = clf_test(leftGlcm_test_feature, leftGlcm_test_image_name,
                                 leftGlcm_model_path, leftGlcm_scaler_path, "left_glcm")
    rightGlcm_test_res = clf_test(rightGlcm_test_feature, rightGlcm_test_image_name,
                                  rightGlcm_model_path, rightGlcm_scaler_path, "right_glcm")

    df_test_result_list.append(geo_test_res)
    df_test_result_list.append(leftFosf_test_res)
    df_test_result_list.append(rightFosf_test_res)
    df_test_result_list.append(leftGlcm_test_res)
    df_test_result_list.append(rightGlcm_test_res)
    df_test_result = reduce(lambda left, right: pd.merge(left, right, on=['image_name']),
                            df_test_result_list)
    df_ensemble_test_features = pd.concat((df_test_result['geo_anomaly_score'],
                                        df_test_result['left_fosf_anomaly_score'],
                                        df_test_result['right_fosf_anomaly_score'],
                                        df_test_result['left_glcm_anomaly_score'],
                                        df_test_result['right_glcm_anomaly_score']), axis=1)
    ensemble_test_features = df_ensemble_test_features.values

    ensemble_test_label = geo_test_label
    test_image_name = geo_test_image_name

    ensemble_model = joblib.load(ensemble_model_path)
    pre_label = ensemble_model.predict(ensemble_test_features)
    pre_label[np.where(pre_label == 1)] = 0
    pre_label[np.where(pre_label == -1)] = 1
    anomaly_score = ensemble_model.decision_function(ensemble_test_features)
    acc = accuracy_score(ensemble_test_label, pre_label)
    conf_m = confusion_matrix(ensemble_test_label, pre_label)
    Auc = roc_auc_score(ensemble_test_label, -anomaly_score)
    sensitivity = conf_m[1, 1] / (conf_m[1, 0] + conf_m[1, 1])
    specificity = conf_m[0, 0] / (conf_m[0, 0] + conf_m[0, 1])
    print("Auc:{0}  sensitivity:{1}  specificity:{2}  conf_m:{3}".format(Auc, sensitivity, specificity,conf_m))

