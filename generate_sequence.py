"""
create on 2019/07
used for generate sequence
user_sequence={uid:[seq1,seq2,...]} where,seq=[pid,...]
seq_time={uid:[time1,time2,...]} where time=0/1 for weekend/workday
"""

import argparse
import time
from collections import defaultdict


class sequence(object):
    def __init__(self, file_path):
        user_sequence = defaultdict(list)
        # seq_list=[seq1, seq2, ...]
        seq_list = set()

        seq_time = {}
        time_list = []

        user_dict = {}
        poi_dict = {}
        loc_geo = {}

        with open(file_path, 'r')as f:
            pre_user = -1
            date = '-1'
            user_count = 0
            # seq = [pid, ...]
            seq = []

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

                wday = time.strptime(time_str, '%Y-%m-%d').tm_wday
                wday = 0 if wday < 5 else 1
                rating = 1
                if poi not in poi_dict:
                    # 若poi_id 不在poi_dict，那么不在poi_dict={poi_id:第几个key}
                    poi_dict[poi] = poi_dict.get(poi, int(len(poi_dict)))

                poi = poi_dict[poi]

                if poi not in loc_geo:
                    loc_geo[poi] = (lat, lon)

                if user != pre_user:
                    pre_user = user
                    seq_list = []
                    time_list = []
                if date == time_str:
                    seq.append(poi)
                    continue
                else:
                    seq = []
                    date = time_str
                seq.append(poi)
                seq_list.append(seq)
                time_list.append(wday)
                user_sequence[user] = seq_list
                seq_time[user] = time_list
        # print(seq_list)
        for user in user_sequence.keys():
            print("user\n {}".format(user))
            print("len:{} user_sequence:\n {}".format(len(user_sequence[user]), user_sequence[user]))
            print("len:{} seq_time:\n {}".format(len(seq_time[user]), seq_time[user]))

        # print(poi_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-train', help='Training file', dest='fi', required=True)
    # path = './data/Gowalla_totalCheckins.txt'
    path = './data/mini_checkins.txt'
    parser.add_argument('-train', help='Training file', dest='file_path', default=path)

    args = parser.parse_args()

    vocab = sequence(args.file_path)
