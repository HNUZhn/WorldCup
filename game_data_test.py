# -*- coding: UTF-8 -*-
import pandas as pd
import requests
from bs4 import BeautifulSoup
from pandas.core.frame import DataFrame
import json

dict = {'受三球':3,'受两球半/三球':2.75,'受两球半':2.5,'受两球/两球半':2.25,'受两球':2,'受球半/两球':1.75,'受球半':1.5,'受一球/球半':1.25,'受一球':1,'受半球/一球':0.75,'受半球':0.5,'受平手/半球':0.25,'平手':0,
	'三球':-3,'两球半/三球':-2.75,'两球半':-2.5,'两球/两球半':-2.25,'两球':-2,'球半/两球':-1.75,'球半':-1.5,'一球/球半':-1.25,'一球':-1,'半球/一球':-0.75,'半球':-0.5,'平手/半球':-0.25,'三球/三球半':-3.25,'三球半':-3.5}
data = []
for ij in range(1,35):
    url = 'http://liansai.500.com/index.php?c=score&a=getmatch&stid=11740&round=%s'%ij
    r = requests.get(url, params={'wd': 'python'})
    soup = BeautifulSoup(r.text, 'lxml')
    info = soup.select('p')
    useful_str = info[0].text.split('[')[1].split(']')[0]
    useful_str2 = useful_str.replace('{', '@{')
    useful_list = useful_str2.split(',@')
    for i in range(len(useful_list)):
        useful_list[i] = useful_list[i].replace('@', '')
    for game_info in useful_list:
        game_info_dic = json.loads(game_info)
        # print(game_info)
        match_date = str(game_info_dic['stime'])  # 比赛日期
        team_home = str(game_info_dic['hname'])  # 主队
        team_visiting = str(game_info_dic['gname'])  # 客队
        game_type = str(game_info_dic['round'])  # 比赛类型
        result = str(game_info_dic['fid'])  # 结果
        big_small_ball_line = str(game_info_dic['color'])  # 大小球盘
        print(dict[game_info_dic['handline'].strip()])
        try:
            hand_score = float(dict[game_info_dic['handline'].strip()])  # 盘后 -1.5表示 球半 -0.25表示平手/半球
        except:
            hand_score = game_info_dic['handline']  # 盘后 -1.5表示 球半 -0.25表示平手/半球
        try:
            win_rate = float(game_info_dic['win'])  # 欧赔平均胜:
            draw_rate = float(game_info_dic['draw'])  # 欧赔平均平
            lost_rate = float(game_info_dic['lost'])  # 欧赔平均负
        except:
            win_rate = game_info_dic['win']  # 欧赔平均胜
            draw_rate = game_info_dic['draw']  # 欧赔平均平
            lost_rate = game_info_dic['lost']  # 欧赔平均负

        home_score = int(game_info_dic['hscore'])  # 主队进球
        visiting_score = int(game_info_dic['gscore'])  # 客队进球

        big_small_ball = 2.5

        # big_small_ball = game_info_dic['BS'] #大小球结果
        total_score = float(home_score + visiting_score)
        win_score = float(home_score - visiting_score)

        result_list = [match_date, team_home, team_visiting, game_type, hand_score, win_rate, draw_rate, lost_rate,
                       big_small_ball,
                       home_score, visiting_score, result, big_small_ball_line, total_score, win_score]

        data.append(result_list)
    print('第%s轮增加成功' % ij)
path = 'data/game_info_test.csv'
frame_data2 = DataFrame(data)
frame_data2.rename(
        columns={0: 'match_date', 1: 'team_home', 2: 'team_visiting', 3: 'game_type', 4: 'hand_score', 5: 'win_rate',
                 6: 'draw_rate', 7: 'lost_rate', 8: 'big_small_ball',
                 9: 'home_score', 10: 'visiting_score', 11: 'result', 12: 'big_small_ball_line', 13: 'total_score',
                 14: 'win_score'}, inplace=True)
frame_data3 = frame_data2.dropna(axis=0, how='any')
frame_data3.to_csv(path, encoding='utf-8', header=None)
