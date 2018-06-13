import requests
from bs4 import BeautifulSoup
import re
import pandas as pd

path_team_rank = 'data/team_rank.csv'
team_rank = pd.read_csv(path_team_rank)
team_rank['2017'] = 0
team_rank['2016'] = 0
team_rank['2015'] = 0
team_rank['2014'] = 0
team_rank['2013'] = 0
team_rank['2012'] = 0
team_rank['2011'] = 0
team_rank['2010'] = 0

team_rank.to_csv(path_team_rank,encoding='utf-8')

path_name = 'data/english_and_chinese_name.csv'
names = pd.read_csv(path_name)
# print(type(names),names['search_name'])
a1 = 0
for name in names['search_name']:
    url = 'http://www.fifa.com/fifa-world-ranking/associations/association=%s/men/index.html'%name
    r = requests.get(url,params={'wd': 'python'})
    soup = BeautifulSoup(r.text,'lxml')
    result = []
    for i in range(6,14):
        a = soup.select('#rnk_%s'%i)
       # print (a)
        for info in a:
            d = []
            b = info.find_all('td')
            for b_i in b[1:3]:
                c =b_i.text
                d.append(c)
            d.append(names.loc[a1,'chinese_name'])
            result.append(d)

    team_rank.loc[a1,'2017'] = result[0][0]
    team_rank.loc[a1,'2016'] = result[1][0]
    team_rank.loc[a1,'2015'] = result[2][0]
    team_rank.loc[a1,'2014'] = result[3][0]
    team_rank.loc[a1,'2013'] = result[4][0]
    team_rank.loc[a1,'2012'] = result[5][0]
    team_rank.loc[a1,'2011'] = result[6][0]
    team_rank.loc[a1,'2010'] = result[7][0]

    print('修改%s排名成功'%names.loc[a1,'chinese_name'])
    a1 = a1 + 1

path_team_info = 'data/team_info.csv'
team_info = pd.read_csv(path_team_info)


team_rank.to_csv(path_team_rank,encoding='utf-8')

# path_history = 'data/historical_record.csv'
# history = pd.read_csv(path_history)
#
# a1 = 0
# for i in history['time']:
#     history.loc[a1,'year'] = i.split('/')[0]
#     a1 = a1 +1
#
# history.to_csv(path_history ,encoding='utf-8')
