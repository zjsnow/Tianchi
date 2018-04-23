import pandas as pd
import re



'''
提取出part1和part2的所有数值特征和中文特征，划分到训练集和测试集中
将训练集中各特征缺失比例高于或等于98%的特征进行去除，测试集中的这些特征也进行去除
再去除训练集中y值不规范的样本
'''

ch_pattern=re.compile(u'[\u4e00-\u9fa5]+')
num_pattern=re.compile(r'\d+')

# 判断字符串是否包含中文u'[\u4e00-\u9fa5]+'
def contain_ch(s,col):
    try:
        if ch_pattern.search(s):
            return True
        else:
            return False
    except Exception as e:
        print("无法进行正则匹配的列：" %col)

#判断字符串是否包含数字
def contain_num(s,col):
    try:
        if num_pattern.search(s):
            return True
        else:
            return False
    except Exception as e:
        print("无法进行正则匹配的列：" % col)

#缺失值的情况
def missing_condition(features):
    missing_count = features.isnull().sum()[features.isnull().sum() > 0].sort_values(ascending=True)  # sum()按列求和，返回一个index为之前列名的Series,sort_values进行排序，ascending=True表示是升序排列
    missing_percent = missing_count / len(features) #计算缺失值占总样本的比例
    drop_count=missing_count[missing_percent>=0.98] #去除缺失比例大于0.98的特征
    drop_list=drop_count.index
    # missing_df = pd.concat([drop_count, missing_percent],join='inner', axis=1, keys=['count', 'percent'])
    # print(missing_df)
    return list(drop_list)

# #读取原始数据
# part_1=pd.read_csv('meinian_round1_data_part1_20180408.txt',delimiter='$')
# part_2=pd.read_csv('meinian_round1_data_part2_20180408.txt',delimiter='$')
# y_train = pd.read_csv('meinian_round1_train_20180408.txt')
# y_test = pd.read_csv('[new] meinian_round1_test_a_20180409.txt')
# y_train.set_index('vid', inplace=True)  # 将vid设为索引
# y_test.set_index('vid', inplace=True)
#
# #提取part1的数值特征
# #Dataframe进行数据重塑造，把vid和table_id相同的特征拼接起来(即把描述同一个人的相同的指标的相同或不同描述值用空格连接起来)
# part1_feature=part_1.pivot_table(index='vid', columns='table_id', values='field_results',aggfunc=lambda x: ' '.join(str(v) for v in x))
# part1_ch_list=[] #part1中所有包含中文的特征
# for col in part1_feature.columns:
#     series=part1_feature.ix[part1_feature[col].notnull(),col]
#     s=series[-1] #以每一列的最后一个值的情况代表这一列是否含中文
#     if contain_ch(s, col) == True:
#         part1_ch_list.append(col)
# part1_num_feature=part1_feature.drop(part1_ch_list,axis=1) #去除part1中所有训练集的中文特征
# part1_train_num=pd.concat([y_train,part1_num_feature],axis=1,join='inner') #part1中的训练集数据
# part1_test_num=pd.concat([y_test,part1_num_feature],axis=1,join='inner') #part1中的测试集数据
# part1_test_num.drop(['收缩压','舒张压','血清甘油三酯','血清高密度脂蛋白','血清低密度脂蛋白'],axis=1,inplace=True) #删除测试集中空的5项指标列
# part1_train_num.to_csv('E:\self_study\\tianchi\Health_AI\data\\test\part1_train_num.csv')
# part1_test_num.to_csv('E:\self_study\\tianchi\Health_AI\data\\test\part1_test_num.csv')
#
# #提取part1的中文特征
# part1_ch_feature=part1_feature.ix[:,part1_ch_list]
# part1_train_ch=pd.concat([y_train,part1_ch_feature],axis=1,join='inner')
# part1_test_ch=pd.concat([y_test,part1_ch_feature],axis=1,join='inner')
# part1_test_ch.drop(['收缩压','舒张压','血清甘油三酯','血清高密度脂蛋白','血清低密度脂蛋白'],axis=1,inplace=True) #删除5项指标列
# part1_train_ch.to_csv('E:\self_study\\tianchi\Health_AI\data\\test\part1_train_ch.csv')
# part1_test_ch.to_csv('E:\self_study\\tianchi\Health_AI\data\\test\part1_test_ch.csv')
#
# #提取part2的数值特征
# part2_feature=part_2.pivot_table(index='vid', columns='table_id', values='field_results',aggfunc=lambda x: ' '.join(str(v) for v in x))
# part2_num_list=[] #part2中所有包含数字的特征
# for col in part2_feature.columns:
#     series=part2_feature.ix[part2_feature[col].notnull(),col]
#     s=series[-1]
#     if contain_num(s, col) == True:
#         part2_num_list.append(col)
# part2_num_feature=part2_feature.ix[:,part2_num_list]
# part2_train_num=pd.concat([y_train,part2_num_feature],axis=1,join='inner')#part2中的训练集数据
# part2_test_num=pd.concat([y_test,part2_num_feature],axis=1,join='inner')#part2中的测试集数据
# part2_test_num.drop(['收缩压','舒张压','血清甘油三酯','血清高密度脂蛋白','血清低密度脂蛋白'],axis=1,inplace=True) #删除测试集中空的5项指标
#
# #提取part2的中文特征
# part2_ch_feature=part2_feature.drop(part2_num_list,axis=1)
# part2_train_ch=pd.concat([y_train,part2_ch_feature],axis=1,join='inner')
# part2_test_ch=pd.concat([y_test,part2_ch_feature],axis=1,join='inner')
# part2_test_ch.drop(['收缩压','舒张压','血清甘油三酯','血清高密度脂蛋白','血清低密度脂蛋白'],axis=1,inplace=True) #删除5项指标列
#
#
# #拼接part1和part2的数值特征
# part2_train_num.drop(['收缩压','舒张压','血清甘油三酯','血清高密度脂蛋白','血清低密度脂蛋白'],axis=1,inplace=True) #删除训练中的5项指标，避免合并时重复
# train_num=pd.concat([part1_train_num,part2_train_num],axis=1)
# test_num=pd.concat([part1_test_num,part2_test_num],axis=1)
# train_num.to_csv('E:\self_study\\tianchi\Health_AI\data\\test\\train_num.csv')
# test_num.to_csv('E:\self_study\\tianchi\Health_AI\data\\test\\test_num.csv')
#
# #拼接part1和part2的中文特征
# part2_train_ch.drop(['收缩压','舒张压','血清甘油三酯','血清高密度脂蛋白','血清低密度脂蛋白'],axis=1,inplace=True) #删除训练中的5项指标，避免合并时重复
# train_ch=pd.concat([part1_train_ch,part2_train_ch],axis=1)
# test_ch=pd.concat([part1_test_ch,part2_test_ch],axis=1)
# train_ch.to_csv('E:\self_study\\tianchi\Health_AI\data\\test\\train_ch.csv')
# test_ch.to_csv('E:\self_study\\tianchi\Health_AI\data\\test\\test_ch.csv')


#去除缺失值大于0.98的数值特征
train_num=pd.read_csv('E:\self_study\\tianchi\Health_AI\data\\test\\train_num.csv')
test_num=pd.read_csv('E:\self_study\\tianchi\Health_AI\data\\test\\test_num.csv')
train_ch=pd.read_csv('E:\self_study\\tianchi\Health_AI\data\\test\\train_ch.csv')
test_ch=pd.read_csv('E:\self_study\\tianchi\Health_AI\data\\test\\test_ch.csv')
x_train_num=train_num.ix[:,6:] #所有数值特征
y_train_num=train_num.ix[:,:6] #vid和5项指标
drop_list=missing_condition(x_train_num)
x_train_num.drop(drop_list,axis=1,inplace=True) #去除缺失比例大于0.98的特征
print(len(x_train_num.columns)) #剩余310个数值特征
train_data_num=pd.concat([y_train_num,x_train_num],axis=1) #训练集数字特征
#测试集也去除这些特征
x_test_num=test_num.ix[:,1:]
id_test_num=test_num.ix[:,0]
x_test_num=x_test_num.ix[:,x_train_num.columns] #保留与训练集相同的特征
test_data_num=pd.concat([id_test_num,x_test_num],axis=1) #测试集数字特征

#去除缺失值大于0.98的中文特征
x_train_ch=train_ch.ix[:,6:] #所有中文特征
y_train_ch=train_ch.ix[:,:6] #vid和5项指标
drop_list=missing_condition(x_train_ch)
x_train_ch.drop(drop_list,axis=1,inplace=True) #去除缺失比例大于0.98的特征
print(len(x_train_ch.columns)) #剩余187个中文特征
train_data_ch=pd.concat([y_train_ch,x_train_ch],axis=1) #训练集中文特征
#测试集的处理
x_test_ch=test_ch.ix[:,1:]
id_test_ch=test_ch.ix[:,0]
x_test_ch=x_test_ch.ix[:,x_train_ch.columns] #保留与训练集相同的特征
test_data_ch=pd.concat([id_test_ch,x_test_ch],axis=1) #测试集中文特征

#去除y值不规范的训练样本,49个
drop_list=[]
y_train=y_train_num.ix[:,1:] #提取出5项指标的值
print(y_train.ix[:5,:])
for i in range(len(y_train)):
    try:
        y_train.ix[i,:].astype(float)
    except Exception as e:
        drop_list.append(i)
print(len(drop_list))
train_data_num.drop(drop_list,inplace=True)
train_data_ch.drop(drop_list,inplace=True)

#输出到文件
train_data_num.to_csv('E:\self_study\\tianchi\Health_AI\data\\test\\train_num_uncleaning.csv',index=False)
test_data_num.to_csv('E:\self_study\\tianchi\Health_AI\data\\test\\test_num_uncleaning.csv',index=False)
train_data_ch.to_csv('E:\self_study\\tianchi\Health_AI\data\\test\\train_ch_uncleaning.csv',index=False)
test_data_ch.to_csv('E:\self_study\\tianchi\Health_AI\data\\test\\test_ch_uncleaning.csv',index=False)

