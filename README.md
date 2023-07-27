# Bipolar Disorder
2023年睿抗机器人开发者大赛(RAICOM)任务应用赛初赛----双相障碍检测
## 基本思路
对于少样本、多特征数据的处理思路主要是特征工程，过于复杂的模型很容易导致其过拟合，为扩充可以采用添加高斯噪声和综合采样；双模态特征各自具有不同的分布和医学意义，因此，分别对各特征进行皮尔逊相关系数筛选，再按照相关算法进行特征的融合是比较合理的方法；模型选择，因为数据样本很少，所以可以采用LogisticRegression、SVC等简单模型，然后进行交叉验证；模型融合，使用软投票的方法融合模型，但要注意过拟合现象。数据决定上限，模型决定下限。

## 高斯噪声
通过添加噪声，扩充样本
```
# 添加高斯噪声扩充数据集
def add_gaussian_noise(X, y, noise_stddev):
    num_samples, num_features = X.shape
    noise = np.random.normal(loc=0.0, scale=noise_stddev, size=(num_samples, num_features))
    X_noisy = X + noise
    X_augmented = np.vstack((X, X_noisy))
    y_augmented = np.vstack((y, y))
    return X_augmented, y_augmented

# 设置高斯噪声标准差
noise_stddev = 0.05

```

## 综合采样
进行综合采样，解决样本数据不平衡
```
from imblearn.combine import SMOTETomek
kos = SMOTETomek(random_state=42)  # 综合采样
X_kos, y_kos = kos.fit_sample(augmented_X, augmented_y)
print('综合采样后，训练集 y_kos 中的分类情况：{}'.format(y_kos.value_counts()))
```
## 模型融合
使用软投票对模型进行融合
```
from sklearn.model_selection import GridSearchCV,KFold
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score 

def search_model(X_train, y_train,X_val,y_val, model_save_path):
    """
    创建、训练、优化和保存深度学习模型
    :param X_train, y_train: 训练集数据
    :param X_val,y_val: 验证集数据
    :param save_model_path: 保存模型的路径和名称
    """

    #创建监督学习模型 以决策树为
    
    dt = tree.DecisionTreeClassifier(random_state=42)
#     gs = naive_bayes.GaussianNB()
    svc = svm.SVC(random_state=42,probability=True)
    lr = LogisticRegression(random_state=42)
    gs = naive_bayes.GaussianNB()
    voting_clf = VotingClassifier(estimators=[('lr', lr), ('svc', svc),('gs',gs)], voting='soft')

     
    # 创建一个fbeta_score打分对象 以F-score为例
    scorer = make_scorer(fbeta_score, beta=1)

    # 在分类器上使用网格搜索，使用'scorer'作为评价函数
    kfold = KFold(n_splits=10) #切割成十份

#     # 同时传入交叉验证函数
#     grid_obj = GridSearchCV(voting_clf, lr_params, scorer, cv=kfold)

    #绘制学习曲线
    plot_learning_curve(voting_clf, X_train, y_train, cv=kfold, n_jobs=4)


    # 使用没有调优的模型做预测
    predictions = (voting_clf.fit(X_train, y_train)).predict(X_val)

    scores = cross_val_score(voting_clf, X_train, y_train, cv = 10, scoring = scorer)
    joblib.dump(voting_clf, model_save_path)



    # 汇报调参前和调参后的分数
    print("\nUnoptimized model\n------")
    print("Accuracy score on validation data: {:.4f}".format(accuracy_score(y_val, predictions)))
    print("Recall score on validation data: {:.4f}".format(recall_score(y_val, predictions)))
    print("F-score on validation data: {:.4f}".format(fbeta_score(y_val, predictions, beta = 1)))

```
## 结果
### 验证集
Accuracy score on validation data: 0.8571
Recall score on validation data: 1.0000
F-score on validation data: 0.8571
### 测试集
Accuracy on test data: 0.8750
Recall on test data: 1.0000
F-score on test data: 0.8571
