import pandas as pd
import matplotlib as plt
import re
import pandas as pd
import requests
from bs4 import BeautifulSoup
import json

path_history = 'data/historical_record.csv'
history = pd.read_csv(path_history)

path_big_small = 'data/big_small_ball.csv'
big_small_ball = pd.read_csv(path_big_small)

path_asia_gaming = 'data/asia_gaming.csv'
asia_gaming = pd.read_csv(path_asia_gaming)

path_europe_gaming = 'data/asia_gaming.csv'
europe_gaming = pd.read_csv(path_europe_gaming)

path_competition_process = 'data/path_competition_process.csv'
competition_process = pd.read_csv(path_europe_gaming)

# result = {'俄罗斯':'rus'}
# b1 = 0
# c1 = 0
# home = '俄罗斯'
#
# home_url = 'http://www.fifa.com/fifa-world-ranking/associations/association=%s/men/index.html'%result[home]
# r = requests.get(home_url,params={'wd': 'python'})
# soup = BeautifulSoup(r.text,'lxml')
# result_home = []
# i = 2023 - history.loc[b1,'year']
# a = soup.select('#rnk_%s'%i)
#            # print (a)
# for info in a:
#     d = []
#     b = info.find_all('td')
#     for b_i in b[1:3]:
#         c =b_i.text
#         d.append(c)
#         result_home.append(d)
#
#     print('%s:增加主队%s排名%s成功'%(d,d,d))


http_team_30_home = 'http://liansai.500.com/index.php?c=teams&a=ajax_fixture&hoa=1&records=40&tid=15'
r = requests.get(http_team_30_home,params={'wd': 'python'})
soup = BeautifulSoup(r.text,'lxml')
info = soup.select('p')
useful_str = info[0].text.split('[')[1].split(']')[0]
useful_str2 = useful_str.replace('{','@{')
useful_list = useful_str2.split(',@')
for i in range(len(useful_list)):
    useful_list[i] = useful_list[i].replace('@','')
for game_info in useful_list:
    game_info_dic = json.loads(game_info)
    match_date = game_info_dic['MATCHDATE'] #比赛日期
    team_home = game_info_dic['HOMETEAMSXNAME'] #主队
    team_visiting = game_info_dic['AWAYTEAMSXNAME'] #客队
    game_type = game_info_dic['SIMPLEGBNAME'] #比赛类型
    hand_score = game_info_dic['HANDICAPLINE'] #盘后 -1.5表示 球半 -0.25表示平手/半球
    win_rate = game_info_dic['WIN'] #欧赔平均胜
    draw_rate = game_info_dic['DRAW'] #欧赔平均平
    lost_rate = game_info_dic['LOST'] #欧赔平均负
    home_score = game_info_dic['HOMESCORE'] #主队进球
    visiting_score = game_info_dic['AWAYSCORE'] #客队进球
    result = game_info_dic['RESULT'] #结果
    big_small_ball_line = game_info_dic['HANDINAME'] #大小球盘
    big_small_ball = sum(float(i) for i in game_info_dic['HANDINAME'].split('/')) / len(game_info_dic['HANDINAME'].split('/'))
    # big_small_ball = game_info_dic['BS'] #大小球结果
    result_list = [match_date,team_home,team_visiting,game_type,hand_score,win_rate,draw_rate,lost_rate,home_score,visiting_score,result,big_small_ball_line,big_small_ball]
    print(result_list)









