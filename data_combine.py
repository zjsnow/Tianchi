import pandas as pd
'''将数值特征和中文特征整合
'''

train_data_num=pd.read_csv('E:\self_study\\tianchi\Health_AI\data\\test\\train_data_num_final.csv')
test_data_num=pd.read_csv('E:\self_study\\tianchi\Health_AI\data\\test\\test_data_num_final.csv')
train_data_ch=pd.read_csv('E:\self_study\\tianchi\Health_AI\data\\test\\train_data_ch_final.csv')
test_data_ch=pd.read_csv('E:\self_study\\tianchi\Health_AI\data\\test\\test_data_ch_final.csv')

x_train_num=train_data_num.ix[:,6:]
x_train_ch=train_data_ch.ix[:,6:]
y_train=train_data_ch.ix[:,:6]

x_test_num=test_data_num.ix[:,1:]
x_test_ch=test_data_ch.ix[:,1:]
id_test=test_data_num.ix[:,0]

print(x_train_num.columns)

train_data=pd.concat([y_train,x_train_num,x_train_ch],axis=1)
test_data=pd.concat([id_test,x_test_num,x_test_ch],axis=1)
print(train_data.columns)
train_data.to_csv('E:\self_study\\tianchi\Health_AI\data\\test\\train_data_num_ch.csv',index=False)
test_data.to_csv('E:\self_study\\tianchi\Health_AI\data\\test\\test_data_num_ch.csv',index=False)