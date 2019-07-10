import argparse
import pickle
import time
from collections import defaultdict

import numpy as np


# 每一个项包括词（user/poi)、计数（count)、路径、哈夫曼编码、item_set（poi集合）
class VocalItem:
    def __init__(self, word):
        self.word = word
        self.count = 0
        # Path (list of indices) from the root to the word (leaf)
        self.path = None
        # Huffman encoding
        self.code = None
        self.user_set = {}
        # item_set={poi:rating}
        self.item_set = {}
        self.combination = {}


"""
1.user_dict是对user进行重新编号的的过程 将user与原来check_ins里面的user以字典的形式进行对应 一个user到len的对应，感觉是一个简单的hash 
user_dict={'0': 0, '1': 1, '2': 2, '17': 3} key是之前check_ins 中的user_id，value 是按照user的排序
2.poi_list是poi的序列信息，根据用户、时间（主要是工作日、非工作日之分）而确定的不同序列集合
3.poi_track={ poi_id:user}和poi_time_trac={poi_id:wday}是按照poi_list的长度来索引用户、星期信息 
4.poi_dict 是一个poi到len的映射，和user_dict基本一致 
5.vocab_4table是一个list，该list的元素为VocabItem 
6.loc_geo是一个以poi为key值，经纬度为value的hashmap 
7.poi_per_track记录了一个用户在某一个时段的poi序列信息 
8.vocab_items是以user为key值，VocabItem为value，其中VocabItem的word是user信息 
9.rating是一个标志信息，1为训练，11为测试
--------------------- 
作者：Stray_Cat_Founder 
来源：CSDN 
原文：https://blog.csdn.net/u013735511/article/details/80158854 
版权声明：本文为博主原创文章，转载请附上博文链接！
"""


# 整个数据集合的封装 从文件中读取user_id,poi,timestamp,(lat,lon)
class Vocab:
    def __init__(self, fi, pair, comb, min_count, percentage):
        # 一个hashmap 即本paper中使用的POI字典
        vocab_items = {}
        vocab_4table = []
        user_dict = {}
        poi_dict = {}
        poi_track = {}
        poi_time_track = {}
        # poi_per_track里面放的是一个一个的poi
        # poi_list=[[poi_per_track_1],[poi_per_track_2]]
        # poi_track={第1个poi_per_track:user_1,第2个poi_per_track:user_2,]
        # poi_time_track={第1个poi_per_track: wday1,第2个poi_per_track:wday2,..]
        # 这三个是对应的
        poi_list = []
        # (lat,lon)
        loc_geo = {}
        sequence = defaultdict(list)

        rating_count = 0

        # 每个POI的追踪 放在一个list

        with open(fi, 'r')as f:
            poi_per_track = []
            pre_user = -1
            date = '-1'
            user_count = 0
            for line in f.readlines():
                line = line.strip()
                tokens = line.split('\t')
                user_count += 1
                user = tokens[0]
                # user_dict.get 返回指定键的值，如果值不在字典中返回default值
                user_dict[user] = user_dict.get(user, int(len(user_dict)))
                user = user_dict[user]
                # POI 编号
                poi_id = tokens[-1]
                # T 之前是日期 本paper把一天的 轨迹作为一个Sequence
                # time_str 是日期
                time_str = tokens[1].split('T')[0]
                lat = float(tokens[2])
                lon = float(tokens[3])
                # 年月日转到星期几 输出值在0-6
                wday = time.strptime(time_str, '%Y-%m-%d').tm_wday
                # weekday:0 weekend:1
                wday = 0 if wday < 5 else 1
                rating = 1

                # 如果星期几不一样或者用户不是前一个用户

                if date != time_str or user != pre_user:

                    if len(poi_per_track) != 0:
                        poi_track[len(poi_list)] = pre_user
                        poi_time_track[len(poi_list)] = wday
                        poi_list.append(poi_per_track)
                    pre_user = user
                    date = time_str
                    poi_per_track = []

                if poi_id not in poi_dict:
                    # 若poi_id 不在poi_dict，那么不在poi_dict={poi_id:第几个key}
                    poi_dict[poi_id] = poi_dict.get(poi_id, int(len(poi_dict)))
                    vocab_4table.append(VocalItem(poi_id))
                # 把顺序号赋值给poi_id
                poi_id = poi_dict[poi_id]

                if poi_id not in loc_geo:
                    loc_geo[poi_id] = (lat, lon)
                # vocab_4table 里面存放的单元是：VocalItem，VocalItem是一个包含很多信息的项
                vocab_4table[poi_id].count += 1

                poi_per_track.append(poi_id)
                # poi_list.append(poi_list)
                sequence[user] = poi_list
                vocab_items[user] = vocab_items.get(user, VocalItem(user))
                vocab_items[user].item_set[poi_id] = rating
                rating_count += 1

                # if rating_count % 10000 == 0:
                #     sys.stdout.write("\rReading ratings %d" % rating_count)
                #     sys.stdout.flush()

            print("{} reading completed!".format(fi))

            self.vocab_items = vocab_items
            self.poi_dict = poi_dict
            self.rating_count = rating_count
            self.user_count = len(user_dict.keys())
            self.item_count = len(poi_dict.keys())
            self.poi_track = poi_track
            self.poi_list = poi_list
            self.poi_time_track = poi_time_track
            self.vocab_4table = vocab_4table
            self.loc_geo = loc_geo
            self.test_data = {}
            self.user_dict = user_dict
            self.poi_dict = poi_dict

            print('len={} \n sequence: {}'.format(len(sequence), sequence))
            # print('vocab_items: {}'.format(self.vocab_items))
            # print('user_dict: {}'.format(self.user_dict))
            # print('poi_dict: {}'.format(self.poi_dict))
            # print('len={} \n poi_dict: {}'.format(len(self.poi_dict),self.poi_dict))
            print('len={} \n poi_track: {}'.format(len(self.poi_track), self.poi_track))
            print('len={} \n poi_list: {}'.format(len(self.poi_list), self.poi_list))
            print('len={} \n poi_time_track: {}'.format(len(self.poi_time_track), self.poi_time_track))

            # print('vocab_4table: {}'.format(self.vocab_4table))
            # print('loc_geo: {}'.format(self.loc_geo))
            # print('test_data: {}'.format(self.test_data))

            # print('num_poi_track: {}'.format(len(self.poi_track)))
            # print('num_poi_time_track: {}'.format(len(self.poi_time_track)))

            # with open("./data/gowalla_loc.hash", 'w')as f:
            #     for x in self.poi_dict:
            #         f.write(str(x) + '\t' + str(self.poi_dict[x]) + '\n')

            print("self.rating_count={}".format(self.rating_count))
            # print("self.rating_count * percentage={}".format(self.rating_count * percentage))

            self.split(percentage)

            print("Total user in training file: {}".format(self.user_count))
            print("Total item in training file: {}".format(self.item_count))
            print("Total rating in file: {}".format(self.rating_count))
            print("Total POI track (day) in file: {}".format(len(self.poi_track.keys())))
            print("Total POI sequence in file: {}".format(len(poi_list)))

    """
    该算法是通过随机数来选择一个user中的poi，并将该poi标记为测试数据。之后将测试数据从训练数据中移除
    """

    def split(self, percentage):
        cur_test = 0

        test_case = int((1 - percentage) * self.rating_count)
        print('Test case: ', test_case)
        # print test_case
        for user in self.vocab_items.keys():
            if len(self.vocab_items[user].item_set.keys()) < 15:
                continue
            if cur_test >= test_case:
                break
            for item in self.vocab_items[user].item_set.keys():
                if cur_test < test_case and np.random.random() > percentage:
                    cur_test += 1
                    self.vocab_items[user].item_set[item] += 10
        seq_out = './data/gowalla_loc.seq.out'
        seq = './data/seq.out'
        # with open(seq_out, 'w')as f:
        #     for i in range(len(self.poi_track)):
        #         try:
        #             user = self.poi_track[i]
        #             w_day = self.poi_time_track[i]
        #         except:
        #             print(self.poi_track[i])
        #             print(self.poi_time_track[i])
        #         if user not in self.test_data:
        #             self.test_data[user] = {}
        #             self.test_data[user][0] = []
        #             self.test_data[user][1] = []
        #         for item in self.poi_list[i]:
        #             if item in self.vocab_items[user].item_set.keys():
        #                 f.write(str(item) + ' ')
        #                 if self.vocab_items[user].item_set[item] > 8:
        #                     self.test_data[user][w_day].append(item)
        #                     self.poi_list[i].remove(item)
        #         f.write('\n')

        with open(seq, 'wb')as f:
            pickle.dump(self.vocab_items, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-train', help='Training file', dest='fi', required=True)
    # path = './data/Gowalla_totalCheckins.txt'
    path = './data/mini_checkins.txt'
    parser.add_argument('-train', help='Training file', dest='fi', default=path)
    parser.add_argument('-pair', help='Pairwise Ranking file', dest='pair')
    parser.add_argument('-comb', help='Combination file', dest='comb')
    parser.add_argument('-split', help='Split for testing', dest='split', type=float, default=0.8)
    parser.add_argument('-model', help='Output model file', dest='fo', default='test')
    parser.add_argument('-cbow', help='1 for CBOW, 0 for skip-gram', dest='cbow', default=0, type=int)
    parser.add_argument('-negative',
                        help='Number of negative examples (>0) for negative sampling, 0 for hierarchical softmax',
                        dest='neg', default=5, type=int)
    parser.add_argument('-dim', help='Dimensionality of word embeddings', dest='dim', default=50, type=int)
    parser.add_argument('-alpha', help='Starting alpha', dest='alpha', default=0.05, type=float)
    parser.add_argument('-beta', help='Starting beta', dest='beta', default=0.1, type=float)
    parser.add_argument('-window', help='Max window length', dest='win', default=5, type=int)
    parser.add_argument('-min-count', help='Min count for words used to learn <unk>', dest='min_count', default=5,
                        type=int)
    parser.add_argument('-processes', help='Number of processes', dest='num_processes', default=15, type=int)
    parser.add_argument('-binary', help='1 for output model in binary format, 0 otherwise', dest='binary', default=0,
                        type=int)
    parser.add_argument('-num_non_neighbors', help='number of sampled negative samples in bpr',
                        dest='num_non_neighbors', default=10, type=int)
    parser.add_argument('-neighbor_threshold', help='distance of neighbor threshold', dest='neighbor_threshold',
                        type=int, default=5)
    # TO DO: parser.add_argument('-epoch', help='Number of training epochs', dest='epoch', default=1, type=int)

    args = parser.parse_args()

    vocab = Vocab(args.fi, args.pair, args.comb, args.min_count, args.split)
