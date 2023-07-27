# 导入相关库
import warnings
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
from time import time
from minepy import MINE
from sklearn import svm
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn import naive_bayes
from scipy.stats import pearsonr
from sklearn.manifold import TSNE
from IPython.display import display
from datetime import datetime as dt
import joblib
from sklearn.decomposition import PCA
from sklearn.metrics import fbeta_score
from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold
from sklearn.feature_selection import chi2
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split




def data_processing_and_feature_selecting(data_path): 
    """
    特征选择
    :param  data_path: 数据集路径
    :return: new_features,label: 经过预处理和特征选择后的特征数据、类标数据
    """ 
    
     #导入医疗数据
    data_xls = pd.ExcelFile(data_path)
    data={}

#查看数据名称与大小
    for name in data_xls.sheet_names:
        df = data_xls.parse(sheet_name=name,header=None)
        print("%-8s 表的 shape:"%name,df.shape)
        data[name] = df
    
#获取 特征1 特征2 类标    
    feature1_raw = data['Feature1']
    feature2_raw = data['Feature2']
    label = data['label']


    # 初始化一个 scaler，并将它施加到特征上
    scaler = MinMaxScaler()
    feature1 = pd.DataFrame(scaler.fit_transform(feature1_raw))
    feature2 = pd.DataFrame(scaler.fit_transform(feature2_raw))

    # 整合得到新特征
   
    print(feature2_raw.shape)
    features = pd.concat([feature1_raw,feature2_raw],axis=1)
    print(features.shape)
    select_feature_number = 12
    select_features = SelectKBest(lambda X, Y: tuple(map(tuple, np.array(list(map(lambda x: pearsonr(x, Y), X.T))).T)),
                                  k=select_feature_number).fit(features,np.array(label).flatten()).get_support(indices=True)
    print(select_features)

    # 特征选择
    new_features = features[features.columns.values[select_features]]
    

    # 返回筛选后的数据
    return new_features,label


    
# -------------------------- 请加载您最满意的模型 ---------------------------
# 加载模型(请加载你认为的最佳模型)
model_path = 'D:\\code\\python\\数据挖掘\\双目障碍检测\\results\\my_model.m'

# 加载模型
model = joblib.load(model_path)

# ---------------------------------------------------------------------------

def predict(new_features):
    """
    加载模型和模型预测
    :param  new_features : 测试数据
    :return y_predict : 预测结果
    """

    y_predict = model.predict(new_features)

    
    # 返回图片的类别
    return y_predict

if __name__=="__main__":
    new_features,label=data_processing_and_feature_selecting("D:\code\python\数据挖掘\双目障碍检测\DataSet.xlsx")
    print(new_features)
    print(predict(np.array(new_features)))