#查看CPU
import os
print('CPU核心数为:',os.cpu_count())
#读入数据并查看数据
import pandas as pd
data=pd.read_csv('/home/aistudio/data/data74924/data.csv')
#查看数据集信息
print('数据集类型:',type(data))
print('数据集规模:',data.shape)
data.info()
pd.isnull(data)
pd.isnull(data).sum()    #统计数据空缺值
pd.isnull(data).sum()    #统计数据空缺值
#查看数据统计属性
data.describe()
import matplotlib.pyplot as plt     #载入绘图库
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
data_name1=['concave points_worst','perimeter_worst','concave points_mean','radius_worst']
def viz_box(data_name):
    box_data=[]    #依据输入字符串生成数据列表
    for name in data_name :
        box_data.append(data[str(name)])
    fig1=plt.figure()    #开始绘图
    ax1=fig1.add_subplot(111)
    ax1.set_title('Boxplot data')
    ax1.boxplot(box_data,sym='r+',showmeans=True)
    ax1.set_xlabel('target')
    ax1.set_ylabel('Data')
    ax1.set_xticks(np.arange(1,5))
    ax1.set_xticklabels(data_name)
    plt.tight_layout()
    plt.show()
viz_box(data_name1)
list_name2=['perimeter_mean','area_worst','radius_mean','area_mean']
viz_box(list_name2)
#数据编码
from sklearn.preprocessing import LabelEncoder
Lb=LabelEncoder()
data['diagnosis-C']=Lb.fit_transform(data['diagnosis'])
print(data['diagnosis-C'].value_counts())    #统计数据类型
print(data['diagnosis'].value_counts())
import seaborn as sns
sns.distplot(data['diagnosis-C'],hist=True,kde=False,rug=True)    #绘制数据类别散点图
import matplotlib.pyplot as plt
fig1=plt.figure()
ax1=fig1.add_subplot(111)
#ax1.set_title('diagnosis-C')
ax1.grid(color='black',linestyle='-.',alpha=0.3)
ax1.hist(data['diagnosis-C'])
ax1.set_xlabel('diagnosis')
ax1.set_ylabel('frequence')
ax1.set_xticks(range(2))
ax1.set_xticklabels(['B','M'])
plt.show()
sns.kdeplot(data['diagnosis-C'],shade=True)    #绘制数据类别核密度估计图
import seaborn as sns
sns.heatmap(data_corr)    #绘制数据相关性热力图
sns.clustermap(data_corr)
data_scatter_matrix=['diagnosis-C','concave points_worst','perimeter_worst','concave points_mean','radius_worst','perimeter_mean','area_worst','radius_mean','area_mean']
sns.pairplot(data[data_scatter_matrix])    #绘制数据相关散点图
scatter_list=['concave points_worst','perimeter_worst','concave points_mean','radius_worst','perimeter_mean','area_worst','radius_mean','area_mean']
y_data=data['diagnosis-C']
i=1
for x_data in scatter_list :
   fig_i=plt.figure()
   ax_i=fig_i.add_subplot(111)
   ax_i.grid(color='black',linestyle='-.',alpha=0.3)
   ax_i.scatter(data[x_data],y_data,color='blue')
   ax_i.set_xlabel(str(x_data))
   ax_i.set_ylabel('diagnosis')
   plt.show()
   i=i+1
x_list=['concave points_worst','perimeter_worst','concave points_mean','radius_worst',]
y_list=['perimeter_mean','area_worst','radius_mean','area_mean']
i=1
for x_data,y_data in zip(x_list,y_list) :
   fig_i=plt.figure()
   ax_i=fig_i.add_subplot(111)
   ax_i.scatter(data.loc[data['diagnosis-C']==0,str(x_data)],data.loc[data['diagnosis-C']==0,str(y_data)],color='blue',label='B')
   ax_i.scatter(data.loc[data['diagnosis-C']==1,str(x_data)],data.loc[data['diagnosis-C']==1,str(y_data)],color='red',label='M')
   ax_i.set_xlabel(str(x_data))
   ax_i.set_ylabel(str(y_data))
   ax_i.legend()
   plt.tight_layout()
   plt.show()
   i=i+1
sns.scatterplot(x='concave points_worst',y='perimeter_worst',hue='diagnosis',data=data)    #绘制数据散点图
sns.scatterplot(x='concave points_mean',y='radius_worst',hue='diagnosis',data=data)    #绘制数据散点图
sns.scatterplot(x='perimeter_mean',y='area_worst',hue='diagnosis',data=data)    #绘制数据散点图
sns.scatterplot(x='radius_mean',y='area_mean',hue='diagnosis',data=data)    #绘制数据散点图
sns.jointplot(x='concave points_worst',y='perimeter_worst',data=data)    #绘制数据散点图
sns.jointplot(x='concave points_mean',y='radius_worst',data=data)    #绘制数据散点图
sns.jointplot(x='perimeter_mean',y='area_worst',data=data)    #绘制数据散点图
sns.jointplot(x='radius_mean',y='area_mean',data=data)    #绘制数据散点图
sns.distplot(data['concave points_worst'],hist=True,kde=True,rug=True)
sns.distplot(data['perimeter_worst'],hist=True,kde=True,rug=True)
sns.distplot(data['concave points_mean'],hist=True,kde=True,rug=True)
sns.distplot(data['radius_worst'],hist=True,kde=True,rug=True)
sns.distplot(data['perimeter_mean'],hist=True,kde=True,rug=True)
sns.distplot(data['area_mean'],hist=True,kde=True,rug=True)
sns.distplot(data['area_worst'],hist=True,kde=True,rug=True)
sns.distplot(data['area_mean'],hist=True,kde=True,rug=True)
#3D形式查看数据分布
x_list=['concave points_worst','perimeter_worst','concave points_mean']
y_list=['radius_worst','perimeter_mean','area_worst']
z_list=['radius_mean','area_mean','concavity_mean']
i=1
for x_data,y_data,z_data in zip(x_list,y_list,z_list) :
   fig_i=plt.figure()
   ax_i=fig_i.add_subplot(111,projection='3d')
   #ax_i.set_title(str(x_data)+'/'+str(y_data)+'/'+str(z_data))
   ax_i.grid(color='black',linestyle='-.',alpha=0.3)
   ax_i.scatter(data.loc[data['diagnosis-C']==0,str(x_data)],data.loc[data['diagnosis-C']==0,str(y_data)],data.loc[data['diagnosis-C']==0,str(z_data)],color='blue',label='B')
   ax_i.scatter(data.loc[data['diagnosis-C']==1,str(x_data)],data.loc[data['diagnosis-C']==1,str(y_data)],data.loc[data['diagnosis-C']==1,str(z_data)],color='red',label='M')
   ax_i.set_xlabel(str(x_data))
   ax_i.set_ylabel(str(y_data))
   ax_i.set_zlabel(str(z_data))
   ax_i.legend()
   plt.tight_layout()
   plt.show()
   i=i+1
#划分数据集
X=data[scatter_list]
y=data['diagnosis']
from sklearn.model_selection import train_test_split    #划分数据集
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8,random_state=123)
print('训练集大小:{0},测试集大小:{1}'.format(X_train.shape,X_test.shape))
print('训练集分类:{0},测试集分类:{1}'.format(y_train.value_counts(),y_test.value_counts()))
#标准化数据集
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler,MinMaxScaler
Std=StandardScaler()    #标准化数据集
Std=Std.fit(X_train)
X_train=DataFrame(Std.transform(X_train))
X_test=DataFrame(Std.transform(X_test))
X_train.describe()
data_box=[X_train[0],X_train[1],X_train[2],X_train[3]]
name1=['concave points_worst','perimeter_worst','concave points_mean','radius_worst']
fig2=plt.figure()    #开始绘图
ax2=fig2.add_subplot(211)
ax2.boxplot(data_box,sym='r+',showmeans=True)
ax2.set_xlabel('target')
ax2.set_ylabel('Data')
ax2.set_xticks(np.arange(1,5))
ax2.set_xticklabels(name1)
plt.tight_layout()
plt.show()
data_box=[X_train[4],X_train[5],X_train[6],X_train[7]]    #绘制第二幅图
name1=['perimeter_mean','area_worst','radius_mean','area_mean']
fig2=plt.figure()
ax3=fig2.add_subplot(212)
ax3.boxplot(data_box,sym='r+',showmeans=True)
ax3.set_xlabel('target')
ax3.set_ylabel('Data')
ax3.set_xticks(np.arange(1,5))
ax3.set_xticklabels(name1)
plt.tight_layout()
plt.show()
#训练决策树模型
from sklearn.tree import DecisionTreeClassifier
max_depth=np.arange(1,100)    #训练最佳树深度
train_err_DTC,test_err_DTC=[],[]
for i in max_depth:
    model_DTC=DecisionTreeClassifier(criterion='gini',max_depth=i)
    model_DTC.fit(X_train,y_train)
    train_err_DTC.append((1-model_DTC.score(X_train,y_train)))
    test_err_DTC.append((1-model_DTC.score(X_test,y_test)))
fig3=plt.figure()
ax3=fig3.add_subplot(111)
ax3.grid(color='black',linestyle='-.',alpha=0.3)
ax3.plot(max_depth,train_err_DTC,color='red',linestyle='-',label='Train error')
ax3.plot(max_depth,test_err_DTC,color='blue',linestyle='-',label='Test error')
ax3.set_xlabel('Max_depth')
ax3.set_ylabel('Error')
ax3.legend()
plt.show()
import time
DTC_start=time.perf_counter()
best_depth=max_depth[test_err_DTC.index(np.min(test_err_DTC))]    #提取最佳树深度
model_DTC=DecisionTreeClassifier(max_depth=best_depth)
model_DTC.fit(X_train,y_train)
print('决策树最佳树深度为:',best_depth)
print('决策树运行时间为:',time.perf_counter()-DTC_start)
#网格搜索决策树算法
from sklearn.model_selection import GridSearchCV
params_DTC=[{'max_leaf_nodes':[8,9,10,11,12],'min_samples_leaf':[0,1],'min_samples_split':[2,3,4,5,6]}]
model_DTC_Grid=GridSearchCV(model_DTC,params_DTC,scoring='precision_micro',return_train_score=True,n_jobs=64)
model_DTC_Grid.fit(X_train,y_train)
#提取最优超参数
print('决策树模型最优参数为:',model_DTC_Grid.best_params_)
model_DTC=model_DTC_Grid.best_estimator_
#模型测量
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score,classification_report
def model_evaluate(model,X_train,y_train,X_test,y_test):
    train_pred=model.predict(X_train)    #训练误差测定
    precision_train=precision_score(y_train,train_pred,pos_label='B')
    recall_train=recall_score(y_train,train_pred,pos_label='B')
    f1_train=f1_score(y_train,train_pred,pos_label='B')
    accuracy_train=accuracy_score(y_train,train_pred)
    print('模型训练准确率为:{0},召回率为:{1},f1分数为:{2},精度为:{3}'.format(precision_train,recall_train,f1_train,accuracy_train))
    test_pred=model.predict(X_test)    #测试误差测定
    precision_test=precision_score(y_test,test_pred,pos_label='B')
    recall_test=recall_score(y_test,test_pred,pos_label='B')
    f1_test=f1_score(y_test,test_pred,pos_label='B')
    accuracy_test=accuracy_score(y_test,test_pred)
    print('模型训练准确率为:{0},召回率为:{1},f1分数为:{2},精度为:{3}'.format(precision_test,recall_test,f1_test,accuracy_test))
    train_err=1-model.score(X_train,y_train)
    test_err=1-model.score(X_test,y_test)
    print('模型训练误差:{0},测试误差:{1}'.format(train_err,test_err))
    print(classification_report(y_test,test_pred))
model_evaluate(model_DTC, X_train, y_train, X_test, y_test)
#绘制PR曲线
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_curve
def PR_curve(model,X,y):
    y_copy=y.copy()
    lb=LabelBinarizer()
    y_label=lb.fit_transform(y_copy)    #分类标签编码
    y_score=model.predict_proba(X)[:,1]
    precision,recall,thresholds=precision_recall_curve(y_label, y_score)
    accuracy=accuracy_score(y,model.predict(X))
    fig4=plt.figure()
    ax4=fig4.add_subplot(111)
    #ax4.set_title('PR curve')
    #ax4.grid(color='black',linestyle='-.',alpha=0.3)
    ax4.plot(precision,recall,color='blue',linestyle='-',label='Accuracy=%f'%accuracy)
    ax4.plot([0,1],[1,0],color='red',linestyle='-',alpha=0.3)
    ax4.set_xlabel('Recall')
    ax4.set_ylabel('Precision')
    ax4.legend()
    plt.show()
PR_curve(model_DTC, X_test, y_test)
#定义ROC曲线绘制函数
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve,roc_auc_score
def ROC_curve(model,X,y):
    y_copy=y.copy()
    lb=LabelBinarizer()
    y_label=lb.fit_transform(y_copy)    #分类标签数字化
    y_score=model.predict_proba(X)[:,1]
    fpr,tpr,thresholds=roc_curve(y_label,y_score)
    auc=roc_auc_score(y_copy,y_score)
    fig5=plt.figure()
    ax5=fig5.add_subplot(111)
    #ax5.set_title('ROC Curve')
    #ax5.grid(color='black',linestyle='-.',alpha=0.3)
    ax5.plot(fpr,tpr,color='blue',linestyle='-',label='AUC=%f'%auc)
    ax5.plot([0,1],[0,1],color='red',alpha=0.3)
    ax5.set_xlabel('Fpr')
    ax5.set_ylabel('Tpr')
    ax5.legend()
    plt.show()
ROC_curve(model_DTC, X_test, y_test)
#可视化混淆矩阵
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
def viz_confusion_matrix(model,X,y):
    y_pred=model.predict(X)
    con_matr=confusion_matrix(y, y_pred)
    con_matr=ConfusionMatrixDisplay(con_matr,display_labels=['B','M'])
    con_matr.plot(include_values=True, cmap='viridis', xticks_rotation='horizontal', values_format='d', ax=None)
viz_confusion_matrix(model_DTC, X_train, y_train)    #可视化决策树混淆矩阵(训练)
viz_confusion_matrix(model_DTC, X_test, y_test)

#训练随机森林模型
from sklearn.ensemble import RandomForestClassifier
n=np.arange(1,100)    #训练最佳树量
train_err_forst,test_err_forst=[],[]
for i in n:
    model_forst=RandomForestClassifier(n_estimators=i,criterion='gini',max_depth=12,n_jobs=64)
    model_forst.fit(X_train,y_train)
    train_err_forst.append((1-model_forst.score(X_train,y_train)))
    test_err_forst.append((1-model_forst.score(X_test,y_test)))
fig3=plt.figure()
ax3=fig3.add_subplot(111)
#ax3.set_title('n')
ax3.grid(color='black',linestyle='-.',alpha=0.3)
ax3.plot(n,train_err_forst,color='red',linestyle='-',label='Train error')
ax3.plot(n,test_err_forst,color='blue',linestyle='-',label='Test error')
ax3.set_xlabel('n')
ax3.set_ylabel('Error')
ax3.legend()
plt.show()
Forst_start=time.perf_counter()
best_n=n[test_err_forst.index(np.min(test_err_forst))]
model_forst=RandomForestClassifier(n_estimators=best_n,max_depth=12,criterion='gini')
model_forst.fit(X_train,y_train)
print('随机森林模型基础分类模型为:',best_n)
print('随机森林模型运行时间为:',time.perf_counter()-Forst_start)
model_evaluate(model_forst, X_train, y_train, X_test, y_test)
viz_confusion_matrix(model_forst, X_train, y_train)
viz_confusion_matrix(model_forst, X_test, y_test)
PR_curve(model_forst, X_test, y_test)
ROC_curve(model_forst, X_test, y_test)
#训练K近邻,逻辑回归,SVM,梯度下降,朴素贝叶斯分类模型
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
model_knn=KNeighborsClassifier(n_neighbors=2,n_jobs=64)
model_Log=LogisticRegression(n_jobs=64)
model_svc=SVC()
model_list=[model_knn,model_Log,model_svc]
model_name=['K近邻','Logistic回归','支持向量机']
for model,name in zip(model_list,model_name):
    Model_start=time.perf_counter()
    model.fit(X_train,y_train)
    print('模型运行时间为:',time.perf_counter()-Model_start)
    print('{0}模型评价'.format(name))
    print(model_evaluate(model, X_train, y_train, X_test, y_test))
    viz_confusion_matrix(model, X_train, y_train)
    viz_confusion_matrix(model, X_test, y_test)
#绘制SVM模型PR曲线
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import LabelBinarizer
y_test_copy=y_test.copy()
lb=LabelBinarizer()
y_test_label=lb.fit_transform(y_test_copy)
y_score=cross_val_predict(model_svc, X_test, y_test_label,cv=10, n_jobs=64)
precision_svc,recall_svc,thresholds_svc=precision_recall_curve(y_test_label, y_score)
fig_SVM=plt.figure()
ax_SVM=fig_SVM.add_subplot(111)
ax_SVM.plot(precision_svc,recall_svc,color='orange',linestyle='-',label='accuracy=%f'%accuracy_score(y_test,model_svc.predict(X_test)))
ax_SVM.plot([0,1],[1,0],color='red',linestyle='-',alpha=0.3)
ax_SVM.set_xlabel('Recall')
ax_SVM.set_ylabel('Precision')
ax_SVM.legend()
plt.show()
#绘制SVM模型ROC曲线
from sklearn.metrics import roc_curve,roc_auc_score
y_score_roc=cross_val_predict(model_svc, X_test, y_test_label,cv=10, n_jobs=64)
fpr_svc,tpr_svc,thresholds_svc_roc=roc_curve(y_test_label, y_score)
auc_svc=roc_auc_score(y_test_copy,y_score_roc)
fig_ROC=plt.figure()
ax_ROC=fig_ROC.add_subplot(111)
ax_ROC.plot(fpr_svc,tpr_svc,color='ORANGE',linestyle='-',label='AUC=%f'%auc_svc)
ax_ROC.plot([0,1],[0,1],color='red',linestyle='-',alpha=0.3)
ax_ROC.set_xlabel('Fpr')
ax_ROC.set_ylabel('Tpr')
ax_ROC.legend()
plt.show()
SGD_start=time.perf_counter()
model_Sgd=SGDClassifier(max_iter=10000,tol=1e-3,penalty='l2',eta0=0.1)
model_Sgd.fit(X_train,y_train.ravel())
print('梯度下降分类模型运行时间为:',time.perf_counter()-SGD_start)
print('梯度下降分类模型评价')
model_evaluate(model_Sgd, X_train, y_train, X_test, y_test)
viz_confusion_matrix(model_Sgd, X_train, y_train)
viz_confusion_matrix(model_Sgd, X_test, y_test)
#绘制SGD模型PR曲线
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import LabelBinarizer
y_test_copy=y_test.copy()
lb=LabelBinarizer()
y_test_label=lb.fit_transform(y_test_copy)
y_score=cross_val_predict(model_Sgd, X_test, y_test_label,cv=10, n_jobs=64)
precision_sgd,recall_sgd,thresholds_sgd=precision_recall_curve(y_test_label, y_score)
fig8=plt.figure()
ax8=fig8.add_subplot(111)
ax8.plot(precision_sgd,recall_sgd,color='blue',linestyle='-',label='accuracy=%f'%accuracy_score(y_test,model_Sgd.predict(X_test)))
ax8.plot([0,1],[1,0],color='red',linestyle='-',alpha=0.3)
ax8.set_xlabel('Recall')
ax8.set_ylabel('Precision')
ax8.legend()
plt.show()
#绘制SGD模型ROC曲线
from sklearn.metrics import roc_curve,roc_auc_score
y_score_roc=cross_val_predict(model_Sgd, X_test, y_test_label,cv=10, n_jobs=64)
fpr_Sgd,tpr_Sgd,thresholds_Sgd=roc_curve(y_test_label, y_score)
auc_log=roc_auc_score(y_test_copy,y_score_roc)
fig9=plt.figure()
ax9=fig9.add_subplot(111)
ax9.plot(fpr_Sgd,tpr_Sgd,color='blue',linestyle='-',label='AUC=%f'%auc_log)
ax9.plot([0,1],[0,1],color='red',linestyle='-',alpha=0.3)
ax9.set_xlabel('Fpr')
ax9.set_ylabel('Tpr')
ax9.legend()
plt.show()
from sklearn.ensemble import AdaBoostClassifier    #AdaBoost算法优化Logistic回归
n_list=np.arange(1,200)
train_err_Log,test_err_Log=[],[]
for n in n_list :
    model_Log_Ada=AdaBoostClassifier(model_Log,n_estimators=i,algorithm='SAMME.R',learning_rate=0.5)
    model_Log_Ada.fit(X_train,y_train)
    train_err_Log.append((1-model_Log_Ada.score(X_train,y_train)))
    test_err_Log.append((1-model_Log_Ada.score(X_test,y_test)))
fig6=plt.figure()
ax6=fig6.add_subplot(111)
ax6.grid(color='black',linestyle='-.',alpha=0.3)
ax6.plot(n_list,train_err_Log,color='red',linestyle='-',label='Train error')
ax6.plot(n_list,test_err_Log,color='blue',linestyle='-',label='Test error')
ax6.set_xlabel('n')
ax6.set_ylabel('Error')
ax6.legend()
plt.show()
model_evaluate(model_Log_Ada, X_train, y_train, X_test, y_test)
viz_confusion_matrix(model_Log_Ada, X_train, y_train)
viz_confusion_matrix(model_Log_Ada, X_test, y_test)

#绘制Ada(Log)模型PR曲线
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import LabelBinarizer
y_test_copy=y_test.copy()
lb=LabelBinarizer()
y_test_label=lb.fit_transform(y_test_copy)
y_score=model_Log_Ada.predict_proba(X_test)[:,1]
precision_Log_Ada,recall_Log_Ada,thresholds_Log_Ada=precision_recall_curve(y_test_label, y_score)
fig4=plt.figure()
ax4=fig4.add_subplot(111)
ax4.plot(precision_Log_Ada,recall_Log_Ada,color='blue',linestyle='-',label='accuracy=%f'%accuracy_score(y_test,model_Log_Ada.predict(X_test)))
ax4.plot([0,1],[1,0],color='red',linestyle='-',alpha=0.3)
ax4.set_xlabel('Recall')
ax4.set_ylabel('Precision')
ax4.legend()
plt.show()
#绘制Ada(Log)模型ROC曲线
from sklearn.metrics import roc_curve,roc_auc_score
y_score_roc=model_Log_Ada.predict_proba(X_test)[:,1]
fpr_Ada_Log,tpr_Ada_Log,thresholds_roc_Ada_Log=roc_curve(y_test_label, y_score)
auc_log=roc_auc_score(y_test_copy,y_score_roc)
fig5=plt.figure()
ax5=fig5.add_subplot(111)
ax5.plot(fpr_Ada_Log,tpr_Ada_Log,color='blue',linestyle='-',label='AUC=%f'%auc_log)
ax5.plot([0,1],[0,1],color='red',linestyle='-',alpha=0.3)
ax5.set_xlabel('FPR')
ax5.set_ylabel('TPR')
ax5.legend()
plt.show()
#绘制多个模型PR曲线
from sklearn.model_selection import cross_val_predict
mode_list=[model_DTC,model_forst,model_knn,model_Log,model_svc]
model_name=['DecisionTree','RandomForst','KNN','Logistic']
color_list=['blue','yellow','green','purple']
list_i=np.arange(4)
y_test_copy=y_test.copy()    #分类标签数字化
lb=LabelBinarizer()
y_test_label=lb.fit_transform(y_test_copy)
precision_list,recall_list,thresholds_list,acccuracy_list=[],[],[],[]
for model in model_list:
    y_score=cross_val_predict(model, X_test, y_test_label,cv=10, n_jobs=64)
    precision,recall,thresholds=precision_recall_curve(y_test_label, y_score)
    accuracy=accuracy_score(y_test,model.predict(X_test))
    precision_list.append(precision)
    recall_list.append(recall)
    thresholds_list.append(thresholds)
    acccuracy_list.append(accuracy)
fig6=plt.figure()
ax6=fig6.add_subplot(111)
for i,name,color in zip(list_i,model_name,color_list):
    ax6.plot(precision_list[i],recall_list[i],color=color,linestyle='-',label=str(name)+' '+'accuracy=%f'%acccuracy_list[i])
ax6.plot(precision_svc,recall_svc,color='orange',linestyle='-',label='SVM accuracy=%f'%accuracy_score(y_test,model_svc.predict(X_test)))
ax6.plot(precision_sgd,recall_sgd,color='black',linestyle='-',label='SGD accuracy=%f'%accuracy_score(y_test,model_Sgd.predict(X_test)))
ax6.plot(precision_Log_Ada,recall_Log_Ada,color='red',linestyle='-',label='AdaBoost(Logistic) accuracy=%f'%accuracy_score(y_test,model_Log_Ada.predict(X_test)))
ax6.plot([0,1],[1,0],color='red',linestyle='-',alpha=0.3)
ax6.set_xlabel('Recall')
ax6.set_ylabel('Precision')
ax6.legend()
plt.show()
#绘制多个模型ROC曲线
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.model_selection import cross_val_predict
mode_list=[model_DTC,model_forst,model_knn,model_Log,model_GNB,model_Sgd,model_Log_Ada]
model_name=['DecisionTree','RandomForst','KNN','Logistic','GuassBayesian','SGD','AdaBoost-logistic']
color_list=['cyan','blue','black','green','orange','purple','grey']
list_i=np.arange(4)
y_test_copy=y_test.copy()    #分类标签数字化
lb=LabelBinarizer()
y_test_label=lb.fit_transform(y_test_copy)
fpr_list,tpr_list,thresholds_roc_list,auc_list=[],[],[],[]
for model in model_list:
    y_score=cross_val_predict(model, X_test, y_test_label,cv=10, n_jobs=64)
    fpr,tpr,thresholds=roc_curve(y_test_label, y_score)
    auc=roc_auc_score(y_test,y_score)
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    thresholds_roc_list.append(thresholds)
    auc_list.append(auc)
fig7=plt.figure()
ax7=fig7.add_subplot(111)
for i,name,color in zip(list_i,model_name,color_list):
    ax7.plot(fpr_list[i],tpr_list[i],color=color,linestyle='-',label=str(name)+'AUC=%f'%auc_list[i])
ax7.plot(fpr_svc,tpr_svc,color='ORANGE',linestyle='-',label='SVM AUC=%f'%auc_log)
ax7.plot(fpr_Sgd,tpr_Sgd,color='purple',linestyle='-',label='SGD AUC=%f'%auc_log)
ax7.plot(fpr_Ada_Log,tpr_Ada_Log,color='red',linestyle='-',label='AdaBoost(Logistic) AUC=%f'%auc_log)
ax7.plot([0,1],[0,1],color='red',linestyle='-',alpha=0.5)
ax7.set_xlabel('FPR')
ax7.set_ylabel('TPR')
ax7.legend()
plt.show()
#计算模型推理时间
Model_list=[model_DTC,model_forst,model_knn,model_Log,model_svc,model_GNB,model_Sgd,model_Log_Ada]
for model in Model_list:
    Time_start=time.perf_counter()
    Predict_Data=model.predict(X_test)
    Run_time=time.perf_counter()-Time_start
    Run_model=Run_time/X_test.shape[0]
    print(str(model)+'运行时间为:',Run_model)
#数据准备
data_copy=pd.read_csv('/home/aistudio/data/data74924/data.csv')
Lb=LabelEncoder()
data_copy['diagnosis-C']=Lb.fit_transform(data_copy['diagnosis'])
X_copy=data_copy[scatter_list]
y_copy=data_copy['diagnosis']
Std=StandardScaler()    #标准化数据集
Std=Std.fit(X_copy)
X_copy=DataFrame(Std.transform(X_copy))
data_pred=model_Log_Ada.predict(X_copy)
data_copy['diagnosis-pred-C']=Lb.fit_transform(data_pred)
#使用AdaBoos集成Logistic回归绘制数据散点图
fig10=plt.figure()
ax10=fig10.add_subplot(111)
ax10.scatter(data_copy.loc[data_copy['diagnosis-C']==0,'concave points_worst'],data_copy.loc[data_copy['diagnosis-C']==0,'perimeter_worst'],color='red',marker='o',alpha=0.5,label='B(true)')
ax10.scatter(data_copy.loc[data_copy['diagnosis-C']==1,'concave points_worst'],data_copy.loc[data_copy['diagnosis-C']==1,'perimeter_worst'],color='blue',marker='o',alpha=0.5,label='M(true)')
ax10.scatter(data_copy.loc[data_copy['diagnosis-pred-C']==0,'concave points_worst'],data_copy.loc[data_copy['diagnosis-pred-C']==0,'perimeter_worst'],color='blue',marker='+',label='M(predict)')
ax10.scatter(data_copy.loc[data_copy['diagnosis-pred-C']==1,'concave points_worst'],data_copy.loc[data_copy['diagnosis-pred-C']==1,'perimeter_worst'],color='red',marker='+',label='M(predict)')
ax10.set_xlabel('concave points_worst')
ax10.set_ylabel('perimeter_worst')
ax10.legend()
plt.show()
fig11=plt.figure()
ax11=fig11.add_subplot(111,projection='3d')
#ax11.set_title('True/Pred data')
ax11.scatter(data_copy.loc[data_copy['diagnosis-C']==0,'concave points_worst'],data_copy.loc[data_copy['diagnosis-C']==0,'perimeter_worst'],data_copy.loc[data_copy['diagnosis-C']==0,'concave points_mean'],color='red',marker='o',alpha=0.3,label='B(true)')
ax11.scatter(data_copy.loc[data_copy['diagnosis-C']==1,'concave points_worst'],data_copy.loc[data_copy['diagnosis-C']==1,'perimeter_worst'],data_copy.loc[data_copy['diagnosis-C']==1,'concave points_mean'],color='blue',marker='o',alpha=0.3,label='M(true)')
ax11.scatter(data_copy.loc[data_copy['diagnosis-pred-C']==0,'concave points_worst'],data_copy.loc[data_copy['diagnosis-pred-C']==0,'perimeter_worst'],data_copy.loc[data_copy['diagnosis-pred-C']==0,'concave points_mean'],color='blue',marker='+',label='M(predict)')
ax11.scatter(data_copy.loc[data_copy['diagnosis-pred-C']==1,'concave points_worst'],data_copy.loc[data_copy['diagnosis-pred-C']==1,'perimeter_worst'],data_copy.loc[data_copy['diagnosis-pred-C']==1,'concave points_mean'],color='red',marker='+',label='M(predict)')
ax11.set_xlabel('concave points_worst')
ax11.set_ylabel('perimeter_worst')
ax11.set_zlabel('concave points_mean ')
ax11.legend()
plt.show()
#保存模型
import joblib
mode_list=['model_DTC','model_forst','model_knn','model_Log','model_Sgd','model_Log_Ada']
try :
    for model in mode_list:
        joblib.dump(eval(model),str(model)+'.pkl')
    print('模型保存成功!')
except :
    print('模型保存失败!')