"""
create on 2019/07
used for generate sequence
user_sequence={uid:[seq1,seq2,...]} where,seq=[pid,...]
seq_time={uid:[time1,time2,...]} where time=0/1 for weekend/workday
"""

import time


class sequence(object):
    def __init__(self, file_path):
        user_sequence = {}
        # seq_list=[seq1, seq2, ...]
        seq_list = []
        # seq = [pid, ...]
        seq = []
        seq_time = {}
        time_list = []

        user_dict = {}

        with open(file_path, 'r')as f:
            pre_user = -1
            date = '-1'
            user_count = 0

            for line in f.readlines():
                records = line.split('\t')
                user_count += 1
                user = records[0]
                time_str = records[1].split('T')[0]
                hour = records[1].split('T')[1].split(':')[0]
                lat = float(records[2])
                lon = float(records[3])
                poi = records[-1]
                # user_dict.get 返回指定键的值，如果值不在字典中返回default值
                user_dict[user] = user_dict.get(user, int(len(user_dict)))
                user = user_dict[user]

                wday=time.strptime(time_str,'%Y-%m-%d').tm_wday
                wday = 0 if wday < 5 else 1
                rating = 1

