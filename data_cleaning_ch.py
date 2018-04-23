import pandas as pd
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import re
import random
import math

'''
清洗训练集和测试集中的中文特征，目前只处理部分
'''

#观察中文特征有哪些不同的种类
def kind_num(x_train,col_list):
    m=len(x_train)
    for col in col_list:
        kind_dict =dict()
        for i in range(m):
            s=x_train.ix[i,col]
            if s not in kind_dict.keys():
                kind_dict[s] = 1
            else:
                kind_dict[s] = kind_dict[s] + 1
        print(col,OrderedDict(sorted(kind_dict.items(), key=lambda t: t[1],reverse=True)))

#观察数值特征特殊的表达种类
def special_num(x_train,col):
    kind_list=[]
    for i in range(len(x_train)):
        try:
            float(x_train.ix[i, col])
        except Exception as e:
            if x_train.ix[i, col] is not np.nan:
                kind_list.append(x_train.ix[i, col])
    print(col,set(kind_list))

def medical_history(s): #根据有无病史进行划分
    if s is np.nan:
        return '无病史'
    elif '病史' in s:
        return "病史"
    else:
        return '无病史'

def swollen(s):  #淋巴结是否肿大，分为缺失、肿大，正常三种情况
    if s is np.nan:
        return '缺失'
    elif '未触及' in s:
        return '缺失'
    elif '肿大' in s and '无肿大' not in s and '不肿大' not in s:
        return '肿大'
    else:
        return '正常'

def feature_0912(s): #甲状腺分为缺失（不确定），正常，结节和肿大四种情况
    if s is np.nan:
        return np.nan
    elif '不确定' in s:
        return np.nan
    elif '异常' in s or '不肿大' in s:
        return '正常'
    elif '结节' in s:
        return '结节'
    else:
        return '肿大'

def normal(s): #缺失则填缺失的，包含正常或未见异常字眼是正常的，其他说明是异常的
    if s is np.nan:
        return '缺失'
    elif ('正常' in s and '低于正常'not in s) or '未见异常' in s or '未见明显异常' in s:
        return '正常'
    else:
        return '异常'

def yinyang(s): #分为阴性、阳性和缺失三种情况
    s=str(s)
    if '阴性' in s:
        return '阴性'
    elif '阳性' in s:
        return '阳性'
    else:
        return '缺失'

def feature_4001(s): #血管特征分为缺失、正常和异常三种情况
    if s is np.nan:
        return '缺失'
    elif '正常' in s or '未见异常' in s or '良好' in s:
        return '正常'
    else:
        return '异常'

def median(series):
    num_list = []
    for s in series:
        try:
            s = float(s)
            if math.isnan(s) == False:  # 判断是否是nan
                num_list.append(s)
        except Exception as e:
            continue
    # print(num_list)
    return  np.percentile(num_list, 50)

def get_num(s,median):  #数值特征，获取数字,缺失值不用处理
    try:
        float(s)
    except Exception as e:
        s=re.findall(r'\d+\.?\d*',s) #找到所有的数字
        if s==[]: #没有找到
            s=np.nan
        else:
            s = np.mean([float(i) for i in s])  # 取平均值
    finally:
        return s

def feature_30007(s):
    if s is np.nan:
        return np.nan
    elif 'Ⅰ' in s:
        return 1
    elif 'Ⅱ' in s or 'Ⅱ度' in s or 'II' in s or '中度' in s or '正常' in s or '未见异常' in s:
        return 2
    elif 'Ⅲ' in s or 'Ⅲ度' in s:
        return 3
    elif 'Ⅳ' in s or 'Ⅳ度' in s:
        return 4
    else:
        return np.nan

#处理训练集的中文特征
train_data=pd.read_csv('E:\self_study\\tianchi\Health_AI\data\\test\\train_ch_uncleaning.csv')
x_train=train_data.ix[:,6:]
y_train=train_data.ix[:,:6]
train_columns=x_train.columns
important_columns=['0409','0911','0912','1001','1316','1402','3301','4001','1363','1873','300036','30007'] #所有进行处理的特征
#中文特征
kind_num(x_train,['30007'])
ch_columns=['0409','0911','0912','1001','1316','1402','3301','4001']
x_train["0409"]=x_train["0409"].map(medical_history)
x_train['0911']=x_train['0911'].map(swollen)
x_train['0912']=x_train['0912'].map(feature_0912)
x_train[['1001','1316','1402']]=x_train[['1001','1316','1402']].applymap(normal)
x_train['3301']=x_train['3301'].map(yinyang)
x_train['4001']=x_train['4001'].map(feature_4001)
x_train_encoding=pd.get_dummies(x_train[ch_columns])
#数值特征
median_1363=median(x_train['1363'])
x_train['1363']=x_train['1363'].map(lambda s:get_num(s,median_1363))
median_1873=median(x_train['1873'])
x_train['1873']=x_train['1873'].map(lambda s:get_num(s,median_1873))
median_300036=median(x_train['300036'])
x_train['300036']=x_train['300036'].map(lambda s:get_num(s,median_300036))
x_train['30007']=x_train['30007'].map(feature_30007)
#只保留进行了处理的important_columns
x_train=x_train[['1363','1873','300036','30007']]
print(x_train.ix[:5,:])
print(y_train)
train_data=pd.concat([y_train,x_train,x_train_encoding],axis=1)

train_data.to_csv('E:\self_study\\tianchi\Health_AI\data\\test\\train_data_ch_final.csv',index=False)

# #处理测试集的中文特征
# test_data=pd.read_csv('E:\self_study\\tianchi\Health_AI\data\\test\\test_ch_uncleaning.csv')
# x_test=test_data.ix[:,1:]
# id_test=test_data.ix[:,0]
# test_columns=x_test.columns
# #做训练集同样的处理
# # important_columns=['0409','0911','0912','1001','1316','1402','3301','4001','1363','1873','300036','30007'] #所有进行处理的特征
# #中文特征
# ch_columns=['0409','0911','0912','1001','1316','1402','3301','4001']
# x_test["0409"]=x_test["0409"].map(medical_history)
# x_test['0911']=x_test['0911'].map(swollen)
# x_test['0912']=x_test['0912'].map(feature_0912)
# x_test[['1001','1316','1402']]=x_test[['1001','1316','1402']].applymap(normal)
# x_test['3301']=x_test['3301'].map(yinyang)
# x_test['4001']=x_test['4001'].map(feature_4001)
# x_test_encoding=pd.get_dummies(x_test[ch_columns])
# #数值特征
# median_1363=median(x_test['1363'])
# x_test['1363']=x_test['1363'].map(lambda s:get_num(s,median_1363))
# median_1873=median(x_test['1873'])
# x_test['1873']=x_test['1873'].map(lambda s:get_num(s,median_1873))
# median_300036=median(x_test['300036'])
# x_test['300036']=x_test['300036'].map(lambda s:get_num(s,median_300036))
# x_test['30007']=x_test['30007'].map(feature_30007)
# # 只保留进行了处理的important_columns
# x_test=x_test[['1363','1873','300036','30007']]
# test_data=pd.concat([id_test,x_test,x_test_encoding],axis=1)
# test_data.to_csv('E:\self_study\\tianchi\Health_AI\data\\test\\test_data_ch_final.csv',index=False)
