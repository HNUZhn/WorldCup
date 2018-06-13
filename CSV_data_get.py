import re
import pandas as pd
import requests
from bs4 import BeautifulSoup

path_history = 'data/historical_record.csv'
history = pd.read_csv(path_history)
history['total_score'] = 0
history['loss_score'] = 0
Data_in = history.loc[history['home_rank']!=0]
Data = Data_in.loc[history['visiting_rank']!=0]
result = Data['result']

for i in result:
    total_score = int(i.split('-')[0]) + int(i.split('-')[1])
    loss_score = int(i.split('-')[0]) - int(i.split('-')[1])
    # print ("比分和%s，比分差%s"%(total_score,loss_score))

print(len(history),len(Data_in),len(Data))
i_ = j_=  l_  = 0
i1_ = j1_=  l1_  = 0
i2_ = j2_=  l2_  = 0
i3_ = j3_=  l3_  = 0
k = m =0

for index in Data.index:
    if int(Data.loc[index,'home_rank']) < int(Data.loc[index,'visiting_rank']):
        data_info_result = Data.loc[index,'result']
        total_score = int(data_info_result.split('-')[0]) + int(data_info_result.split('-')[1])
        loss_score = int(data_info_result.split('-')[0]) - int(data_info_result.split('-')[1])

        if loss_score>0:
            i_=  i_+1
        elif loss_score==0:
            j_ = j_+1
        else:
            l_ = l_+1
    else:
        data_info_result = Data.loc[index, 'result']
        total_score = int(data_info_result.split('-')[0]) + int(data_info_result.split('-')[1])
        loss_score = int(data_info_result.split('-')[0]) - int(data_info_result.split('-')[1])
        if loss_score < 0:
            i1_ = i1_ + 1
        elif loss_score == 0:
            j1_ = j1_ + 1
        else:
            l1_ = l1_ + 1

    data_info_result = Data.loc[index, 'result']
    total_score = int(data_info_result.split('-')[0]) + int(data_info_result.split('-')[1])
    loss_score = int(data_info_result.split('-')[0]) - int(data_info_result.split('-')[1])
    Data.loc[index, 'total_score'] = total_score
    Data.loc[index, 'loss_score'] = loss_score

    if int(Data.loc[index, 'home_rank']) < int(Data.loc[index, 'visiting_rank']):
        if loss_score > 0:
            i3_ = i3_ + 1
        elif loss_score == 0:
            j3_ = j3_ + 1
        else:
            l3_ = l3_ + 1
    else:
        if loss_score < 0:
            i3_ = i3_ + 1
        elif loss_score == 0:
            j3_ = j3_ + 1
        else:
            l3_ = l3_ + 1

    if loss_score > 0:
        i2_ = i2_ + 1
    elif loss_score == 0:
        j2_ = j2_ + 1
    else:
        l2_ = l2_ + 1

    if abs(loss_score)<=1:
        k = k+1
    if total_score<= 2:
        m = m +1

Data.to_csv(path_history ,encoding='utf-8')

print('主队排名高赢球%s场赢球率%.2f%%，平局%s场，输场数%s输球率%.2f%%，共%s场'%(i_,100*i_/(i_+j_+l_),j_,l_,100*l_/(i_+j_+l_),i_+j_+l_))
print('客队排名高赢球%s场赢球率%.2f%%，平局%s场，输场数%s输球率%.2f%%，共%s场'%(i1_,100*i1_/(i1_+j1_+l1_),j1_,l1_,100*l1_/(i1_+j1_+l1_),i1_+j1_+l1_))
print('主队赢球%s场赢球率%.2f%%，平局%s场，输场数%s输球率%.2f%%，共%s场'%(i2_,100*i2_/(i2_+j2_+l2_),j2_,l2_,100*l2_/(i2_+j2_+l2_),i2_+j2_+l2_))
print('排名高赢球%s场赢球率%.2f%%，平局%s场，输场数%s输球率%.2f%%，共%s场'%(i3_,100*i3_/(i3_+j3_+l3_),j3_,l3_,100*l3_/(i3_+j3_+l3_),i3_+j3_+l3_))
print('净胜球小于等于1球的场数%s,占比%.2f%%'%(k,100*k/2001))
print('0比0的场次%s'%m)