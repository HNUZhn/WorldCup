# -*- coding: UTF-8 -*-

import re
import pandas as pd
import requests
from bs4 import BeautifulSoup
from pandas.core.frame import DataFrame
import csv
import json

path = 'data/game_data.csv'
data = []
##http://liansai.500.com/team/39/teamfixture/

home_url = 'http://liansai.500.com/paiming/'
r = requests.get(home_url, params={'wd': 'python'})
soup = BeautifulSoup(r.text, 'lxml')
a = soup.select('div.ltab_bd_wrap > div > table > tbody')
# dict_result = {'0-0':0,'0-1':1,'0-2':2,'0-3':3,'0-4':4,'0-5':5,'0-6':6,'0-7':7,'0-8':8,'0-9':9,'0-10':10,'1-0':11,'1-1':12,'1-2':13,'1-3':14,'1-4':15,'1-5':16,'1-6':17,'1-7':18,'1-8':19,'1-9':20,'1-10':21,'2-0':22,'2-1':23,'2-2':24,'2-3':25,'2-4':26,'2-5':27,'2-6':28,'2-7':29,'2-8':30,'2-9':31,'2-10':32,'3-0':33,'3-1':34,'3-2':35,'3-3':36,'3-4':37,'3-5':38,'3-6':39,'3-7':40,'3-8':41,'3-9':42,'3-10':43,'4-0':44,'4-1':45,'4-2':46,'4-3':47,'4-4':48,'4-5':49,'4-6':50,'4-7':51,'4-8':52,'4-9':53,'4-10':54,'5-0':55,'5-1':56,'5-2':57,'5-3':58,'5-4':59,'5-5':60,'5-6':61,'5-7':62,'5-8':63,'5-9':64,'5-10':65,'6-0':66,'6-1':67,'6-2':68,'6-3':69,'6-4':70,'6-5':71,'6-6':72,'6-7':73,'6-8':74,'6-9':75,'6-10':76,'7-0':77,'7-1':78,'7-2':79,'7-3':80,'7-4':81,'7-5':82,'7-6':83,'7-7':84,'7-8':85,'7-9':86,'7-10':87,'8-0':88,'8-1':89,'8-2':90,'8-3':91,'8-4':92,'8-5':93,'8-6':94,'8-7':95,'8-8':96,'8-9':97,'8-10':98,'9-0':99,'9-1':100,'9-2':101,'9-3':102,'9-4':103,'9-5':104,'9-6':105,'9-7':106,'9-8':107,'9-9':108,'9-10':109,'10-0':110,'10-1':111,'10-2':112,'10-3':113,'10-4':114,'10-5':115,'10-6':116,'10-7':117,'10-8':118,'10-9':119,'10-10':120}
# dict_result = {'0-0':0,'0-1':1,'0-2':2,'0-3':3,'0-4':4,'0-5':5,'0-6':6,'0-7':7,'1-0':8,'1-1':9,'1-2':10,'1-3':11,'1-4':12,'1-5':13,'1-6':14,'1-7':15,'2-0':16,'2-1':17,'2-2':18,'2-3':19,'2-4':20,'2-5':21,'2-6':22,'2-7':23,'3-0':24,'3-1':25,'3-2':26,'3-3':27,'3-4':28,'3-5':29,'3-6':30,'3-7':31,'4-0':32,'4-1':33,'4-2':34,'4-3':35,'4-4':36,'4-5':37,'4-6':38,'4-7':39,'5-0':40,'5-1':41,'5-2':42,'5-3':43,'5-4':44,'5-5':45,'5-6':46,'5-7':47,'6-0':48,'6-1':49,'6-2':50,'6-3':51,'6-4':52,'6-5':53,'6-6':54,'6-7':55,'7-0':56,'7-1':57,'7-2':58,'7-3':59,'7-4':60,'7-5':61,'7-6':62,'7-7':63}
# dict_result = {'0-0':0,'0-1':1,'0-2':2,'0-3':3,'0-4':4,'0-5':5,'1-0':6,'1-1':7,'1-2':8,'1-3':9,'1-4':10,'1-5':11,'2-0':12,'2-1':13,'2-2':14,'2-3':15,'2-4':16,'2-5':17,'3-0':18,'3-1':19,'3-2':20,'3-3':21,'3-4':22,'3-5':23,'4-0':24,'4-1':25,'4-2':26,'4-3':27,'4-4':28,'4-5':29,'5-0':30,'5-1':31,'5-2':32,'5-3':33,'5-4':34,'5-5':35}
#dict_result = {'1-0':0,'2-0':1,'2-1':2,'3-0':3,'3-1':4,'3-2':5,'4-0':6,'4-1':7,'4-2':8,'4-3':9,'5-0':10,'5-1':11,'5-2':12,'5-3':13,'5-4':14,'0-0':15,'1-1':16,'2-2':17,'3-3':18,'4-4':19,'5-5':20,'0-1':21,'0-2':22,'1-2':23,'0-3':24,'1-3':25,'2-3':26,'0-4':27,'1-4':28,'2-4':29,'3-4':30,'0-5':31,'1-5':32,'2-5':33,'3-5':34,'4-5':35}
dict_result = {'1-0':1,'2-0':4,'2-1':5,'3-0':9,'3-1':10,'3-2':11,'4-0':16,'4-1':17,'4-2':18,'4-3':19,'0-0':0,'1-1':3,'2-2':8,'3-3':15,'4-4':24,'0-1':2,'0-2':6,'1-2':7,'0-3':12,'1-3':13,'2-3':14,'0-4':20,'1-4':21,'2-4':22,'3-4':23}
for info in a:
    data_info = []
    data_use = info.find_all('tr')
    for data_td in data_use:
        data_td_i = data_td.find_all('td')
        team_rank = data_td_i[0].text
        team_name = data_td_i[1].text
        team_http = data_td_i[1].find_all('a')[0].get('href') + 'teamfixture/'
        team_id = team_http.split('/')[4]
        print (team_rank,team_name,team_http,team_id)
        data_info = [team_rank,team_name,team_http,team_id]
        data.append(data_info)
frame_data = DataFrame(data)
frame_data.rename(columns={0:'team_rank', 1:'team_name', 2:'team_http',3:'team_id'},inplace=True)
print(frame_data)
frame_data.to_csv(path,encoding='utf-8')

# writer.writerows(data)
# csvfile.close()

path2 = 'data/game_info.csv'
['match_date', 'team_home', 'team_visiting', 'game_type', 'hand_score', 'win_rate', 'draw_rate', 'lost_rate',
                       'home_score', 'visiting_score', 'result', 'big_small_ball_line', 'big_small_ball']
data2 = []
path_data = 'data/game_data.csv'
history = pd.read_csv(path_data)

for team_id in history['team_id']:
    http_team_50_home = 'http://liansai.500.com/index.php?c=teams&a=ajax_fixture&hoa=1&records=50&tid=%s'%team_id
    r = requests.get(http_team_50_home, params={'wd': 'python'})
    soup = BeautifulSoup(r.text, 'lxml')
    info = soup.select('p')
    try:
        useful_str = info[0].text.split('[')[1].split(']')[0]
    except:
        continue
    useful_str2 = useful_str.replace('{', '@{')
    useful_list = useful_str2.split(',@')
    for i in range(len(useful_list)):
        useful_list[i] = useful_list[i].replace('@', '')
    for game_info in useful_list:
        game_info_dic = json.loads(game_info)
        match_date = str(game_info_dic['MATCHDATE'])  # 比赛日期
        team_home = str(game_info_dic['HOMETEAMSXNAME'])  # 主队
        team_visiting = str(game_info_dic['AWAYTEAMSXNAME'])  # 客队
        game_type = str(game_info_dic['SIMPLEGBNAME'])  # 比赛类型
        result = str(game_info_dic['RESULT'])  # 结果
        if result == '胜':
            result = 2
        elif result == '平':
            result = 1
        elif result == '负':
            result = 0
        else:result = 3
        big_small_ball_line = str(game_info_dic['HANDINAME'])  # 大小球盘

        try:
            hand_score = float(game_info_dic['HANDICAPLINE'].strip())  # 盘后 -1.5表示 球半 -0.25表示平手/半球
        except:
            hand_score = game_info_dic['HANDICAPLINE'] # 盘后 -1.5表示 球半 -0.25表示平手/半球
        try:
            win_rate = float(game_info_dic['WIN'])  # 欧赔平均胜:
            draw_rate = float(game_info_dic['DRAW'])  # 欧赔平均平
            lost_rate = float(game_info_dic['LOST'])  # 欧赔平均负
        except:
            win_rate = game_info_dic['WIN']  # 欧赔平均胜
            draw_rate = game_info_dic['DRAW']  # 欧赔平均平
            lost_rate = game_info_dic['LOST']  # 欧赔平均负

        home_score = int(game_info_dic['HOMESCORE'])  # 主队进球
        visiting_score = int(game_info_dic['AWAYSCORE'])  # 客队进球
        score_str = str(home_score)+'-'+str(visiting_score)
        try:
            score_type = int(dict_result[score_str])
        except:
            score_type = 25
        try:
            big_small_ball = sum(float(i) for i in game_info_dic['HANDINAME'].split('/')) / len(
                game_info_dic['HANDINAME'].split('/'))
        except:
            big_small_ball = None
        # big_small_ball = game_info_dic['BS'] #大小球结果
        total_score = int(home_score + visiting_score)
        win_score = int(home_score - visiting_score)

        result_list = [match_date, team_home, team_visiting, game_type, hand_score, win_rate, draw_rate, lost_rate,big_small_ball,
                       home_score, visiting_score, result, big_small_ball_line,total_score,win_score,score_type]

        data2.append(result_list)
    print ('%s增加成功'%team_home)

frame_data2 = DataFrame(data2)
frame_data2.rename(columns={0:'match_date',1: 'team_home',2: 'team_visiting',3: 'game_type', 4:'hand_score', 5:'win_rate', 6:'draw_rate', 7:'lost_rate',8:'big_small_ball',
                       9:'home_score', 10:'visiting_score', 11:'result', 12:'big_small_ball_line',13:'total_score',14:'win_score',15:'score_type'},inplace=True)
frame_data3 = frame_data2.dropna(axis = 0,how = 'any')
frame_data3.to_csv(path2,encoding='utf-8',header = None)