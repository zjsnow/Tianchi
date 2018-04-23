import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import re
import random
import math

'''
清洗训练集和测试集中的数值特征
'''
#观察有哪些特征需要清洗
def cleaning_feature(x_train):
    count = 0
    col_list = []
    fearture_name = x_train.columns
    for col in fearture_name:
        try:
            x_train.ix[:, col].astype(float)
        except Exception as e:
            count = count + 1
            col_list.append(col)
    print(count)  # 训练集总共有168个数值特征需要进行清洗
    return col_list


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

#缺失值的情况
def missing_condition(features):
    missing_count = features.isnull().sum()[features.isnull().sum() > 0].sort_values(ascending=True)  # sum()按列求和，返回一个index为之前列名的Series,sort_values进行排序，ascending=True表示是升序排列
    missing_percent = missing_count / len(features) #计算缺失值占总样本的比例
    drop_count=missing_count[missing_percent>=0.98] #去除缺失比例大于0.98的特征
    drop_list=drop_count.index
    # missing_df = pd.concat([drop_count, missing_percent],join='inner', axis=1, keys=['count', 'percent'])
    # print(missing_df)
    return list(drop_list)

#提取数值特征中的数字
def get_num(s):
    try:
        float(s)
    except Exception as e:
        if s=='未见': #对于未见这个词用0代替
            s=0
        s=re.findall(r'\d+\.?\d*',s) #找到所有的数字
        if s!=[]:
            s=np.mean([float(i) for i in s]) #取平均值
        else:
            s=np.nan
    finally:
        return s

def yanya(s):
    try:
        float(s)
    except Exception as e:
        if s is np.nan:
            return np.nan
        else:
            high_list = ['偏高', '高']
            normal_list = ['正常']  # 正常范围10-21
            for string in high_list:
                if string in s:
                    return '高'
            for string in normal_list:
                if string in s:
                    return "正常"
            s = re.search('\d+\.?\d*', s) #对于没有检查的，匹配不到数字，也会返回np.nan
            if s==None:
                return np.nan
            else:
                s=s.group(0)
    s = float(s)
    if(s<10):
        return "低"
    elif(s>21):
        return '高'
    else:
        return '正常'

def feature_2409(s):
    try:
        float(s)
    except Exception as e:
        s=re.match('\d+\.?\d*',s)
        s=s.group(0)
    finally:
        s=float(s)
        if(s<12):
            return '低'
        elif(s>22):
            return '高'
        else:
            return '正常'

def most_value(series):
    num_list = []
    for s in series:
        try:
            s = float(s)
            if math.isnan(s) == False:  # 判断是否是nan
                num_list.append(s)
        except Exception as e:
            continue
    # print(num_list)
    return stats.mode(num_list)[0][0], np.percentile(num_list, 1), np.mean(num_list)  # 众数，分位数

def feature_0425(s,mode,min,max):
    normal_list=['正常','未见异常','无异常']
    slow_list=['缓慢']
    fast_list=['急促']
    cucao_list=['粗糙']
    try:
        float(s)
    except Exception as e:
        for string in normal_list:
            if string in s:
                return mode
        for string in slow_list:
            if string in s:
                return min
        for string in fast_list:
            if string in s:
                return max
        for string in cucao_list:
            return 21
        s = re.search('\d+\.?\d*', s)  # 对于没有检查的，匹配不到数字，也会返回np.nan
        if s == None:
            return np.nan
        else:
            s = s.group(0)
            return s

def shili(s):
    weak_list=['手动','光感','指数','无光感']
    normal_list=['正常']
    blind_list=['义眼','失明']
    try:
        float(s)
        return s
    except Exception as e:
        for string in weak_list:
            if string in s:
                return 0.05
        for string in normal_list:
            if string in s:
                return random.choice([1.0,1.2,1.5]) #正常则是1.0以上
        for string in blind_list:
            if string in s:
                return 0
        s = re.findall(r'\d+\.?\d*', s)  # 找到所有的数字
        if s != []:
            s = np.mean([float(i) for i in s])  # 取平均值
        else:
            s=np.nan
        return s

def feature_2413(s):
    try:
        float(s)
        return s
    except Exception as e:
        return

def feature_1171(s,mode):
    try:
        float(s)
        return s
    except Exception as e:
        if s=='阴性':
            return mode
        else:
            s = re.search('\d+\.?\d*', s)
            if s!=None:
                s=s.group(0)
                return s
            else:
                return np.nan

def feature_1002(s): #心率,正常范围为60-100
    try:
        float(s)
    except Exception as e:
        if '正常' in s:
            s=2
        else:
            s=re.search(r'\d+(?=次/分)',s)
            if s!=None:
                s=s.group(0)
            else:
                s=np.nan
    finally:
        s=float(s)
        if s is np.nan:
            return 2  #没有则认为是正常的
        elif s<60: #过缓，标记为1
            return 1
        elif s>100: #过快，标记为3
            return 3
        else:     #正常，标记为2
            return 2

def feature_1334(s): #分为缺失，正常，异常三种情况
    normal_list=['正常','未见异常']
    if s is np.nan: #np.nan是float型
        return '缺失'
    for string in normal_list:
        if string in s:
            return '正常'
    return '异常'

#训练集的数据清洗
train_data=pd.read_csv('E:\self_study\\tianchi\Health_AI\data\\test\\train_num_uncleaning.csv')
x_train=train_data.ix[:,6:]
y_train=train_data.ix[:,:6]
print(len(x_train.columns))
col_list=cleaning_feature(x_train) #得到需要清洗的特征名列表

drop_columns=['0214','1104','1335'] #不重要的特征,剩余307个特征
special_columns=['1319','1320','2409','0104','0425','1321','1322','1325','1326','2413','1002','1334'] #进行特殊处理的特征
common_columns=[] #能够提取出数字的普通数值特征
for col in col_list:
    if col not in drop_columns and col not in special_columns:
        common_columns.append(col)

# for col in col_list:
#     if col in drop_columns:
#         special_num(x_train,col)

x_train.drop(drop_columns,axis=1,inplace=True)
x_train['1319']=x_train['1319'].map(yanya).map({'低':1,'正常':2,'高':3}) #右眼压特征的处理，离散化为高，低，正常
x_train['1320']=x_train['1320'].map(yanya).map({'低':1,'正常':2,'高':3}) #左眼眼压
x_train['2409']=x_train['2409'].map(feature_2409).map({'低':1,'正常':2,'高':3}) #有正常的范围，所以离散化为高，低，正常
x_train.ix[x_train['0104']=='心内各结构未见明显异常','0104']=x_train.ix[x_train['0104']!='心内各结构未见明显异常','0104'].median()#用中位数填充
mode_0425,min_0425,max_0425=most_value(x_train['0425'])
x_train['0425']=x_train['0425'].map(lambda s: feature_0425(s,mode_0425,min_0425,max_0425))
x_train.ix[:,['1321','1322','1325','1326']]=x_train.ix[:,['1321','1322','1325','1326']].applymap(shili) #视力的处理
x_train['2413']=x_train['2413'].map(feature_2413)
x_train['1002']=x_train['1002'].map(feature_1002)
x_train['1334']=x_train['1334'].map(feature_1334)
x_train_encoding=pd.get_dummies(x_train[['1334']])
x_train.ix[:,common_columns]=x_train.ix[:,common_columns].applymap(get_num)#取数字，再取平均,map是作用于series的每一个元素，applymap是作用与dataframe的每一个元素
drop_list=missing_condition(x_train) #再一次去除缺失值超过98%的,['300134', '2163']删除
print(drop_list)
drop_list.append('1334') #1334特征进行了one-hot编码，需要删除
x_train.drop(drop_list,axis=1,inplace=True)
print(len(x_train.columns))
train_data=pd.concat([y_train,x_train,x_train_encoding],axis=1)
train_data.to_csv('E:\self_study\\tianchi\Health_AI\data\\test\\train_data_num_final.csv',index=False)


#测试集的数据清洗
test_data=pd.read_csv('E:\self_study\\tianchi\Health_AI\data\\test\\test_num_uncleaning.csv')
x_test=test_data.ix[:,1:]
id_test=test_data.ix[:,0]
test_col_list=cleaning_feature(x_test) #得到需要清洗的特征名列表,测试集有110个数值特征需要清洗
test_drop_columns=[]
test_special_columns=['1319','1320','2409','0104','0425','1321','1322','1325','1326','2413','1002','1334'] #特殊处理的特征
test_common_columns=[]
for col in test_col_list:
    if col in drop_columns:
        test_drop_columns.append(col) #得到测试集需要删除的特征,也是3个
print(len(test_drop_columns))
test_drop_columns.append('300134')
test_drop_columns.append('2163')

for col in test_col_list:
    if col not in test_drop_columns and col not in test_special_columns:
        test_common_columns.append(col)

# #观察待处理的数值特征的特殊数值特征值情况
# for col in col_list:
#     if col not in test_drop_columns:
#         special_num(x_test,col)

x_test.drop(test_drop_columns,axis=1,inplace=True)
x_test['1319']=x_test['1319'].map(yanya).map({'低':1,'正常':2,'高':3}) #右眼压特征的处理，离散化为高，低，正常
x_test['1320']=x_test['1320'].map(yanya).map({'低':1,'正常':2,'高':3}) #左眼眼压
x_test['2409']=x_test['2409'].map(feature_2409).map({'低':1,'正常':2,'高':3}) #有正常的范围，所以离散化为高，低，正常
x_test.ix[x_test['0104']=='心内各结构未见明显异常','0104']=x_test.ix[x_test['0104']!='心内各结构未见明显异常','0104'].median()#用中位数填充
mode_0425,min_0425,max_0425=most_value(x_test['0425'])
x_test['0425']=x_test['0425'].map(lambda s: feature_0425(s,mode_0425,min_0425,max_0425))
x_test.ix[:,['1321','1322','1325','1326']]=x_test.ix[:,['1321','1322','1325','1326']].applymap(shili) #视力的处理
x_test['2413']=x_test['2413'].map(feature_2413)
x_test['1002']=x_test['1002'].map(feature_1002)
x_test['1334']=x_test['1334'].map(feature_1334)
x_test_encoding=pd.get_dummies(x_test[['1334']])
x_test.drop(['1334'],axis=1,inplace=True)
x_test.ix[:,test_common_columns]=x_test.ix[:,test_common_columns].applymap(get_num)#取数字，再取平均,map是作用于series的每一个元素，applymap是作用与dataframe的每一个元素
print(x_train.columns)
test_data=pd.concat([id_test,x_test,x_test_encoding],axis=1)
test_data.to_csv('E:\self_study\\tianchi\Health_AI\data\\test\\test_data_num_final.csv',index=False)







