import argparse
import math
import multiprocessing
import struct
import sys
import time
import warnings
from collections import OrderedDict

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
1.user_hash是对user进行重新编号的的过程 将user与原来check_ins里面的user以字典的形式进行对应 一个user到len的对应，感觉是一个简单的hash 
user_hash={'0': 0, '1': 1, '2': 2, '17': 3} key是之前check_ins 中的user_id，value 是按照user的排序
2.poi_list是poi的序列信息，根据用户、时间（主要是工作日、非工作日之分）而确定的不同序列集合
3.poi_track和poi_time_track是按照poi_list的长度来索引用户、星期信息 
4.vocab_hash 是一个poi到len的映射，和user_hash基本一致 
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
        user_hash = {}
        vocab_hash = {}
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

        rating_count = 0

        # 每个POI的追踪 放在一个list
        poi_per_track = []
        pre_user = -1
        date = '-1'
        user_count = 0

        with open(fi, 'r')as f:
            for line in f.readlines():
                line = line.strip()
                tokens = line.split('\t')
                user_count += 1
                user = tokens[0]
                # user_hash.get 返回指定键的值，如果值不在字典中返回default值
                user_hash[user] = user_hash.get(user, int(len(user_hash)))
                user = user_hash[user]
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

                if poi_id not in vocab_hash:
                    # 若poi_id 不在vocab_hash，那么不在vocab_hash={poi_id:第几个key}
                    vocab_hash[poi_id] = vocab_hash.get(poi_id, int(len(vocab_hash)))
                    vocab_4table.append(VocalItem(poi_id))
                # 把顺序号赋值给poi_id
                poi_id = vocab_hash[poi_id]

                if poi_id not in loc_geo:
                    loc_geo[poi_id] = (lat, lon)
                # vocab_4table 里面存放的单元是：VocalItem，VocalItem是一个包含很多信息的项
                vocab_4table[poi_id].count += 1

                poi_per_track.append(poi_id)
                vocab_items[user] = vocab_items.get(user, VocalItem(user))
                vocab_items[user].item_set[poi_id] = rating
                rating_count += 1

                # if rating_count % 10000 == 0:
                #     sys.stdout.write("\rReading ratings %d" % rating_count)
                #     sys.stdout.flush()

            print("{} reading completed!".format(fi))

            self.vocab_items = vocab_items
            self.vocab_hash = vocab_hash
            self.rating_count = rating_count
            self.user_count = len(user_hash.keys())
            self.item_count = len(vocab_hash.keys())
            self.poi_track = poi_track
            self.poi_list = poi_list
            self.poi_time_track = poi_time_track
            self.vocab_4table = vocab_4table
            self.loc_geo = loc_geo
            self.test_data = {}

            print('num_poi_track: {}'.format(len(self.poi_track)))
            print('num_poi_time_track: {}'.format(len(self.poi_time_track)))

            with open("./data/gowalla_loc.hash", 'w')as f:
                for x in self.vocab_hash:
                    f.write(str(x) + '\t' + str(self.vocab_hash[x]) + '\n')

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
        seq_out = './gowalla_loc.seq.out'
        with open(seq_out, 'w')as f:
            for i in range(len(self.poi_track)):
                try:
                    user = self.poi_track[i]
                    w_day = self.poi_time_track[i]
                except:
                    print(self.poi_track[i])
                    print(self.poi_time_track[i])
                if user not in self.test_data:
                    self.test_data[user] = {}
                    self.test_data[user][0] = []
                    self.test_data[user][1] = []
                for item in self.poi_list[i]:
                    if item in self.vocab_items[user].item_set.keys():
                        f.write(str(item) + ' ')
                        if self.vocab_items[user].item_set[item] > 8:
                            self.test_data[user][w_day].append(item)
                            self.poi_list[i].remove(item)
                f.write('\n')


class UnigramTable:
    """
    A list of indices of tokens in the vocab following a power law distribution,
    used to draw negative samples.
    """

    def __init__(self, vocab):
        vocab_size = len(vocab)
        power = 0.75
        norm = sum([math.pow(t.count, power) for t in vocab])
        # Normalizing constant

        table_size = int(1e8)  # Length of the unigram table
        table = np.zeros(table_size, dtype=np.uint32)

        print('Filling unigram table')
        p = 0
        # Cumulative probability
        i = 0
        for j, unigram in enumerate(vocab):
            p += float(math.pow(unigram.count, power)) / norm
            while i < table_size and float(i) / table_size < p:
                table[i] = j
                i += 1
        self.table = table

    def sample(self, count):
        indices = np.random.randint(low=0, high=len(self.table), size=count)
        return [self.table[i] for i in indices]


def sigmoid(z):
    if z > 6:
        return 1.0
    elif z < -6:
        return 0.0
    else:
        return 1 / (1 + math.exp(-z))


# Init syn0 with random numbers from a uniform distribution on the interval [-0.5, 0.5]/dim
def init_net(dim, vocab_size, user_size):
    tmp = np.random.uniform(low=0.5 / dim, high=2.0 / dim, size=(vocab_size, dim))
    syn0 = np.ctypeslib.as_ctypes(tmp)
    syn0 = multiprocessing.Array(syn0._type_, syn0, lock=False)

    # init syn_user with random number from a uniform distribution
    tmp = np.random.uniform(low=0.5 / dim, high=2.0 / dim, size=(user_size, dim))
    syn_user = np.ctypeslib.as_ctypes(tmp)
    syn_user = multiprocessing.Array(syn_user._type_, syn_user, lock=False)

    t_size = 2
    tmp = np.random.uniform(low=0.5 / dim, high=2.0 / dim, size=(t_size, dim))
    syn_t = np.ctypeslib.as_ctypes(tmp)
    syn_t = multiprocessing.Array(syn_t._type_, syn_t, lock=False)

    # Init syn1 with zeros
    tmp = np.random.uniform(low=0.5 / dim, high=2.0 / dim, size=(t_size, dim))
    syn1 = np.ctypeslib.as_ctypes(tmp)
    syn1 = multiprocessing.Array(syn1._type_, syn1, lock=False)

    return (syn0, syn_user, syn1, syn_t)


def dis(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1

    c = math.sin(lat1) * math.sin(lat2) + math.cos(lat1) * math.cos(lat2) * math.cos(dlon)
    r = 6371
    if c > 1:
        c = 1
    return int(r * math.acos(c))


def prediction(pid):
    start = vocab.user_count / num_processes * pid
    end = vocab.user_count if pid == num_processes - 1 else vocab.user_count / num_processes * (pid + 1)
    c = 0.0
    while start < end:
        user = start
        c += 1.0
        # sys.stdout.write("\r%f" %(c/(end-start)))
        # sys.stdout.flush()
        u_r = 0
        test_case = 0
        for item in vocab.vocab_items[user].item_set:
            if vocab.vocab_items[user].item_set[item] > 8:
                u_r += 1
        if u_r == 0:
            continue
        result[user] = []
        raw_rating = {}
        for item1 in range(vocab.item_count):
            if item1 in vocab.vocab_items[user].item_set:
                if vocab.vocab_items[user].item_set[item1] < 8:
                    continue
            pred = 0
            for i in range(len(syn0[item1])):
                pred += syn0[item1][i] * syn_user[user][i]
            raw_rating[item1] = pred
        ranked_d = OrderedDict(sorted(raw_rating.items(), key=lambda x: x[1]))
        top10 = ranked_d.keys()[::-1][:10]
        top5 = ranked_d.keys()[::-1][:5]
        g5 = 0.0
        g10 = 0.0
        for i in top5:
            if i in vocab.vocab_items[user].item_set:
                g5 += 1
        for i in top10:
            if i in vocab.vocab_items[user].item_set:
                g10 += 1
        ar = []
        ar.append(g5 / 5.0)
        ar.append(g10 / 10.0)
        ar.append(g5 / u_r)
        ar.append(g10 / u_r)
        result[user] = ar
        #    print len(result[user])
    start += 1


def train_process(pid):  # 训练模型
    # print pid
    # Set fi to point to the right chunk of training file
    # 信息分片
    start = int(len(vocab.poi_list) / num_processes * pid)
    end = int(len(vocab.poi_list) if pid == num_processes - 1 else len(vocab.poi_list) / num_processes * (pid + 1))
    # fi.seek(start)
    print('Worker %d beginning training at %d, ending at %d' % (pid, start, end))
    current_item = start
    alpha = starting_alpha
    iter_num = 0
    total_iter = 1
    word_count = 0
    last_word_count = 0
    orig_start = start
    count = 0
    counttime = time.time()
    total = end - start
    # 在当前的分片序列中训练
    while start < end:
        # 训练信息打印
        if count % 5000 == 0:
            print("worker id %d, deal with %d items(total is %d), cost %d, deal percentage %f" % (
                pid, count, total, time.time() - counttime, count * 1.0 / (total)))
            counttime = time.time()
        count += 1
        # sys.stdout.write("%d iter\n" % (iter_num))
        # if iter_num<total_iter and start==end-1:
        #    iter_num+=1
        #    start=orig_start

        # global_word_count.value+=()
        # line = fi.readline().strip()
        # Skip blank lines
        # if not line:
        #    continue
        # print line.split()
        # Init sent, a list of indices of words in line
        # sent = vocab.indices(['<bol>'] + line.split() + ['<eol>'])
        if len(vocab.poi_list[start]) == 0:
            start += 1
            continue
        # word_count+=1
        if word_count % 2000 == 0:
            global_word_count.value += (word_count - last_word_count)
            last_word_count = word_count
            # print global_word_count.value
        # if global_word_count.value%10000==0:
        #                auc, ndcg10,ndcg20=evaluate(vocab, syn_user,syn0)
        #                p5,p10,r5,r10=tr_error(vocab,syn0,syn_user)
        #                sys.stdout.write( "\nProcessing %d, AUC: %f, NDCG@10: %f, NDCG@20: %f" %(global_word_count.value, auc,ndcg10,ndcg20))
        #                sys.stdout.write( "\nProcessing %d, %f %f %f %f" %(global_word_count.value,p5,p10,r5,r10))
        #                sys.stdout.flush()
        # if current_error.value>last_error.value:
        #    break
        # last_error.value=current_error.value
        # 获取当前poi序列
        sent = vocab.poi_list[start]
        # user信息
        c_user = vocab.poi_track[start]
        # week信息
        t_state = vocab.poi_time_track[start]
        # print c_user
        for sent_pos, token in enumerate(sent):  # 枚举
            # 在设置的window之下，随机选择一个窗口大小。上下文信息为前后的信息
            current_win = np.random.randint(1, win + 1)
            context_start = max(sent_pos - current_win, 0)
            context_end = min(sent_pos + current_win + 1, len(sent))
            # 拼接上下文信息
            context = sent[context_start:sent_pos] + sent[sent_pos + 1:context_end]  # Turn into an iterator?
            # print 'context',(context)
            # print 'word', (sent[sent_pos])
            if alpha != 0:
                # CBOW，word2vec的训练
                if cbow:
                    # Compute neu1
                    neu1 = np.mean(np.array([syn0[c] for c in context]), axis=0)
                    assert len(neu1) == dim, 'neu1 and dim do not agree'

                    # Init neu1e with zeros
                    neu1e = np.zeros(dim)

                    # Compute neu1e and update syn1
                    if neg > 0:
                        classifiers = [(token, 1)] + [(target, 0) for target in table.sample(neg)]
                    else:
                        classifiers = zip(vocab[token].path, vocab[token].code)
                    for target, label in classifiers:
                        # print 'CBOW',target,label
                        z = np.dot(neu1, syn1[target])  # 矩阵乘积
                        p = sigmoid(z)
                        g = alpha * (label - p)
                        neu1e += g * syn1[target]  # Error to backpropagate to syn0
                        syn1[target] += g * neu1  # Update syn1

                    # Update syn0
                    for context_word in context:
                        syn0[context_word] += neu1e

                # Skip-gram，word2vec训练
                else:
                    # 对每个词语分别计算
                    for context_word in context:
                        # Init neu1e with zeros
                        neu1e = np.zeros(dim)

                        # Compute neu1e and update syn1
                        if neg > 0:
                            classifiers = [(token, 1)] + [(target, 0) for target in table.sample(neg)]
                        else:
                            classifiers = zip(vocab[token].path, vocab[token].code)
                        for target, label in classifiers:
                            # print 'SG',target,label
                            z = np.dot(syn0[context_word], syn1[target]) + np.dot(syn0[context_word], syn_t[t_state])
                            p = sigmoid(z)
                            g = alpha * (label - p)
                            neu1e += g * syn1[target]  # Error to backpropagate to syn0
                            syn1[target] += g * (syn0[context_word])  # Update syn1
                            syn_t[t_state] += g * (syn0[context_word])
                        # Update syn0
                        syn0[context_word] += neu1e
            # bayesian personalized ranking，计算neighboring POI和non-neighboring POI，利用地理距离来确定偏号关系
            # print 'bpr',len(vocab.vocab_items[c_user].item_set.keys())
            for x in range((len(vocab.vocab_items[c_user].item_set.keys()))):
                neighbor_item = np.random.choice(vocab.vocab_items[c_user].item_set.keys())
                non_neighbors = []
                while len(non_neighbors) < num_non_neighbors:
                    non_neighbor_item = np.random.randint(0, vocab.item_count)
                    while non_neighbor_item in vocab.vocab_items[c_user].item_set.keys():
                        non_neighbor_item = np.random.randint(0, vocab.item_count)
                    non_neighbors.append(non_neighbor_item)
                neighbor_set = []
                non_neighbor_set = []
                for item in non_neighbors:
                    temp_lat1 = vocab.loc_geo[neighbor_item][0]
                    temp_lon1 = vocab.loc_geo[neighbor_item][1]
                    temp_lat2 = vocab.loc_geo[item][0]
                    temp_lon2 = vocab.loc_geo[item][1]
                    if dis(temp_lon1, temp_lat1, temp_lon2, temp_lat2) < neighbor_threshold:
                        neighbor_set.append(item)
                    else:
                        non_neighbor_set.append(item)
                pair_set = []
                if len(neighbor_set) and len(non_neighbor_set):
                    for item_neighbor in neighbor_set:
                        for item_non_neighbor in non_neighbor_set:
                            pair_set.append((neighbor_item, item_neighbor))
                            pair_set.append((item_neighbor, item_non_neighbor))
                else:
                    for item in (neighbor_set + non_neighbor_set):
                        pair_set.append((neighbor_item, item))

                # print 'tt'
                # print syn_user[c_user]
                # print syn0[neighbor_item]
                # 更新向量
                for item in pair_set:
                    neighbor_item = item[0]
                    non_neighbor_item = item[1]
                    p_e = np.dot(syn_user[c_user], syn0[neighbor_item]) - np.dot(syn_user[c_user],
                                                                                 syn0[non_neighbor_item])
                    #            print 'pe', p_e,np.dot(syn_user[c_user],syn0[neighbor_item]),np.dot(syn_user[c_user],syn0[non_neighbor_item])
                    if p_e > 6:
                        bpr_e = 0
                    elif p_e < -6:
                        bpr_e = 1
                    else:
                        bpr_e = np.exp(-p_e) / (1 + np.exp(-p_e))
                    #                print 'bpre', bpr_e
                    syn_user[c_user] += beta * bpr_e * (syn0[neighbor_item] - syn0[non_neighbor_item])
                    syn0[neighbor_item] += beta * bpr_e * (syn_user[c_user])
                    syn0[non_neighbor_item] += beta * bpr_e * (-syn_user[c_user])
                word_count += 1

        #    print 'bpr finished'

        start += 1
        # word_count+=1
        # print start
    # Print progress info
    #  global_word_count.value += (word_count - last_word_count)


def __init_process(*args):
    global vocab, syn0, syn_user, syn1, syn_t, table, cbow, neg, dim, starting_alpha, beta
    global win, num_processes, global_word_count, last_error, current_error, num_non_neighbors, neighbor_threshold, fi

    vocab, syn0_tmp, syn_user_tmp, syn1_tmp, synt_tmp, table, cbow, neg, dim, starting_alpha, beta, win, num_processes, global_word_count, last_error, current_error, num_non_neighbors, neighbor_threshold = args[
                                                                                                                                                                                                              :-1]
    fi = open(args[-1], 'r')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        syn0 = np.ctypeslib.as_array(syn0_tmp)
        syn1 = np.ctypeslib.as_array(syn1_tmp)
        syn_user = np.ctypeslib.as_array(syn_user_tmp)
        syn_t = np.ctypeslib.as_array(synt_tmp)


def __init_evaluation(*args):
    global result
    global syn0, syn_user, syn_t, vocab, num_processes
    vocab, result, syn0_tmp, syn_user_tmp, synt_tmp, num_processes = args[:]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        syn0 = np.ctypeslib.as_array(syn0_tmp)
        syn_user = np.ctypeslib.as_array(syn_user_tmp)
        syn_t = np.ctypeslib.as_array(synt_tmp)


def save(vocab, syn0, fo, binary):
    print('Saving model to', fo)
    dim = len(syn0[0])
    if binary:
        fo = open(fo, 'wb')
        fo.write('{} {}\n'.format(len(syn0), dim))
        fo.write('\n')
        for token, vector in zip(vocab, syn0):
            fo.write('%s ' % token.word)
            for s in vector:
                fo.write(struct.pack('f', s))
            fo.write('\n')
    else:
        fo = open(fo, 'w')
        fo.write('%d %d\n' % (len(syn0), dim))
        for token, vector in zip(vocab, syn0):
            word = token.word
            vector_str = ' '.join([str(s) for s in vector])
            fo.write('%s %s\n' % (word, vector_str))


def predict_parallel(pid):
    # 确定分片位置
    start = vocab.user_count / num_processes * pid
    end = vocab.user_count if pid == num_processes - 1 else vocab.user_count / num_processes * (pid + 1)
    print("predict usr between %d to %d" % (start, end))
    c = 0.0
    count = 0
    counttime = time.time()
    total = end - start
    while start < end:
        # 打印信息
        if count % 5 == 0:
            print("worker id %d, deal with %d items(total is %d), cost %d, deal percentage %f" % (
                pid, count, total, time.time() - counttime, count * 1.0 / (total)))
            counttime = time.time()
        count += 1
        user = start
        # 测试数据
        if user in vocab.test_data:
            # 两个时段的poi信息
            wday_0 = vocab.test_data[user][0]  # 时间为0
            wday_1 = vocab.test_data[user][1]  # 时间为1  分为两个时间序列进行推荐

            raw_rating_0 = {}
            raw_rating_1 = {}
            if len(wday_0):
                if len(wday_1):
                    #  test_case += 1
                    #   确定比例信息
                    rate_0 = len(wday_0) * 1.0 / (len(wday_0) + len(wday_1))
                    # 对每个item进行预测，获取评分值，进而排序，取top-5和top-10
                    for item1 in range(vocab.item_count):
                        if item1 in vocab.vocab_items[user].item_set:
                            if vocab.vocab_items[user].item_set[item1] < 8:
                                continue
                        pred = 0.0
                        for i in range(len(syn0[item1])):
                            pred += syn0[item1][i] * syn_user[user][i] + syn_t[0][i] * syn_user[user][i]
                        raw_rating_0[item1] = pred
                        pred = 0.0
                        for i in range(len(syn0[item1])):
                            pred += syn0[item1][i] * syn_user[user][i] + syn_t[1][i] * syn_user[user][i]
                        raw_rating_1[item1] = pred
                    ranked_0 = OrderedDict(sorted(raw_rating_0.items(), key=lambda x: x[1], reverse=True))  # 排序
                    ranked_1 = OrderedDict(sorted(raw_rating_1.items(), key=lambda x: x[1], reverse=True))
                    top10_0 = ranked_0.keys()[:10]
                    top10_1 = ranked_1.keys()[:10]
                    top5_0 = ranked_0.keys()[:5]
                    top5_1 = ranked_1.keys()[:5]
                    a = int(rate_0 * 10)
                    b = 10 - a
                    top10 = top10_0[:a] + top10_1[:b]
                    a = int(rate_0 * 5)
                    b = 5 - a
                    top5 = top5_0[:a] + top5_1[:b]
                else:
                    #  test_case += 1
                    # rate_0 = len(wday_0) * 1.0/(len(wday_0)+len(wday_1))
                    for item1 in range(vocab.item_count):
                        if item1 in vocab.vocab_items[user].item_set:
                            if vocab.vocab_items[user].item_set[item1] < 8:
                                continue
                        pred = 0.0
                        for i in range(len(syn0[item1])):
                            pred += syn0[item1][i] * syn_user[user][i] + syn_t[0][i] * syn_user[user][i]
                        raw_rating_0[item1] = pred

                    ranked_0 = OrderedDict(sorted(raw_rating_0.items(), key=lambda x: x[1], reverse=True))
                    top10 = ranked_0.keys()[:10]
                    top5 = ranked_0.keys()[:5]
            else:
                if len(wday_1):
                    #  test_case += 1
                    # rate_0 = len(wday_0) * 1.0/(len(wday_0)+len(wday_1))
                    for item1 in range(vocab.item_count):
                        if item1 in vocab.vocab_items[user].item_set:
                            if vocab.vocab_items[user].item_set[item1] < 8:
                                continue
                        pred = 0.0
                        for i in range(len(syn0[item1])):
                            pred += syn0[item1][i] * syn_user[user][i] + syn_t[1][i] * syn_user[user][i]
                        raw_rating_1[item1] = pred
                    ranked_1 = OrderedDict(sorted(raw_rating_1.items(), key=lambda x: x[1], reverse=True))
                    top10 = ranked_1.keys()[:10]
                    top5 = ranked_1.keys()[:5]
            # 测试中的poi信息，作为基准数据
            test_pois = wday_0 + wday_1
            if len(test_pois):
                g5 = 0.0
                g10 = 0.0
                for i in top5:  # 在5个里面选
                    if i in wday_0 + wday_1:
                        g5 += 1.0
                for i in top10:  # 在10个里面选
                    if i in wday_0 + wday_1:
                        g10 += 1.0
                p5 = g5 / 5.0  # 求平均数
                p10 = g10 / 10.0
                r5 = g5 / len(wday_0 + wday_1)
                r10 = g10 / len(wday_0 + wday_1)
                ar = [p5, p10, r5, r10]

                result[user] = ar
                #    print len(result[user])
                ofile1 = open('result1.txt', 'a')
                ofile1.write(str(user) + str(top5) + str(top10) + '\n')
                # print user
                # print top5
                # print top10
        start += 1


def predict(vocab, syn0, syn_user, syn_t):
    test_case = 0
    n_u = 0
    p5 = 0.0
    r5 = 0.0
    p10 = 0.0
    r10 = 0.0

    t1 = time.time()
    for user in vocab.test_data:
        wday_0 = vocab.test_data[user][0]
        wday_1 = vocab.test_data[user][1]
        n_u += 1
        if n_u % 100 == 0:
            t2 = time.time()
            sys.stdout.write("\r%d of %d finished, cost: %d" % (n_u, vocab.user_count, t2 - t1))
            sys.stdout.flush()
            t1 = t2
        raw_rating_0 = {}
        raw_rating_1 = {}
        if len(wday_0):
            if len(wday_1):
                test_case += 1
                rate_0 = len(wday_0) * 1.0 / (len(wday_0) + len(wday_1))
                for item1 in range(vocab.item_count):
                    if item1 in vocab.vocab_items[user].item_set:
                        if vocab.vocab_items[user].item_set[item1] < 8:
                            continue
                    pred = 0.0
                    for i in range(len(syn0[item1])):
                        pred += syn0[item1][i] * syn_user[user][i] + syn_t[0][i] * syn_user[user][i]
                    raw_rating_0[item1] = pred
                    pred = 0.0
                    for i in range(len(syn0[item1])):
                        pred += syn0[item1][i] * syn_user[user][i] + syn_t[1][i] * syn_user[user][i]
                    raw_rating_1[item1] = pred
                ranked_0 = OrderedDict(sorted(raw_rating_0.items(), key=lambda x: x[1], reverse=True))
                ranked_1 = OrderedDict(sorted(raw_rating_1.items(), key=lambda x: x[1], reverse=True))
                top10_0 = ranked_0.keys()[:10]
                top10_1 = ranked_1.keys()[:10]
                top5_0 = ranked_0.keys()[:5]
                top5_1 = ranked_1.keys()[:5]
                a = int(rate_0 * 10)
                b = 10 - a
                top10 = top10_0[:a] + top10_1[:b]
                a = int(rate_0 * 5)
                b = 5 - a
                top5 = top5_0[:a] + top5_1[:b]
            else:
                test_case += 1
                # rate_0 = len(wday_0) * 1.0/(len(wday_0)+len(wday_1))
                for item1 in range(vocab.item_count):
                    if item1 in vocab.vocab_items[user].item_set:
                        if vocab.vocab_items[user].item_set[item1] < 8:
                            continue
                    pred = 0.0
                    for i in range(len(syn0[item1])):
                        pred += syn0[item1][i] * syn_user[user][i] + syn_t[0][i] * syn_user[user][i]
                    raw_rating_0[item1] = pred
                # pred = 0.0
                #             for i in xrange(len(syn0[item1])):
                #  pred += syn0[item1][i]*syn_user[user][i] + syn_t[1][i]*syn_user[user][i]
                # raw_rating_1[item1] = pred
                ranked_0 = OrderedDict(sorted(raw_rating_0.items(), key=lambda x: x[1], reverse=True))
                # ranked_1 = OrderedDict(sorted(raw_rating_1.items(), key=lambda x:x[1], reverse=True))
                top10 = ranked_0.keys()[:10]
                #   top10_1 = ranked_1.keys()[:10]
                top5 = ranked_0.keys()[:5]
        else:
            if len(wday_1):
                test_case += 1
                # rate_0 = len(wday_0) * 1.0/(len(wday_0)+len(wday_1))
                for item1 in range(vocab.item_count):
                    if item1 in vocab.vocab_items[user].item_set:
                        if vocab.vocab_items[user].item_set[item1] < 8:
                            continue
                    pred = 0.0
                    for i in range(len(syn0[item1])):
                        pred += syn0[item1][i] * syn_user[user][i] + syn_t[1][i] * syn_user[user][i]
                    raw_rating_1[item1] = pred
                    # ranked_0 = OrderedDict(sorted(raw_rating_0.items(), key=lambda x:x[1], reverse=True))
                ranked_1 = OrderedDict(sorted(raw_rating_1.items(), key=lambda x: x[1], reverse=True))
                top10 = ranked_1.keys()[:10]
                top5 = ranked_1.keys()[:5]
        test_pois = wday_0 + wday_1
        if len(test_pois):
            g5 = 0.0
            g10 = 0.0
            for i in top5:
                if i in wday_0 + wday_1:
                    g5 += 1.0
            for i in top10:
                if i in wday_0 + wday_1:
                    g10 += 1.0
            p5 += g5 / 5.0
            p10 += g10 / 10.0
            r5 += g5 / len(wday_0 + wday_1)
            r10 += g10 / len(wday_0 + wday_1)
        # t2=time.time()
        # print 'Cost: ', t2-t1
    print('P@5, P@10, R@5, R@10', p5 / test_case, p10 / test_case, r5 / test_case, r10 / test_case)
    ofile = open('result.txt', 'w')
    ofile.write('p@5:' + str(p5 / test_case) + '\n')
    ofile.write('r@5:' + str(r5 / test_case) + '\n')
    ofile.write('p@10:' + str(p10 / test_case) + '\n')
    ofile.write('r@10:' + str(r10 / test_case) + '\n')


def cal_idcg(n):
    idcg = 0
    for i in range(n):
        idcg += 1 / (np.log(i + 2) / np.log(2))
    return (idcg)


def evaluate(vocab, syn_user, syn0):
    cumu_auc = 0
    n_u = 0
    num_t_u = 0
    cumu_ndcg_10 = 0
    cumu_ndcg_20 = 0
    for u in range(vocab.user_count):
        eval_u = {}
        n_test = 0
        n_u += 1
        for i in range(vocab.item_count):
            if i not in vocab.vocab_items[u].item_set.keys() or vocab.vocab_items[u].item_set[i] > 8:
                eval_u[i] = np.dot(syn_user[u], syn0[i])
                if i in vocab.vocab_items[u].item_set and vocab.vocab_items[u].item_set[i] > 8:
                    n_test += 1
        ranked_u = sorted(eval_u.items(), key=lambda x: x[1], reverse=True)
        n_candidate = len(ranked_u)
        hit = 0
        idcg = cal_idcg(10)
        idcg20 = cal_idcg(20)
        dcg = 0
        dcg20 = 0
        num_correct_pairs = 0
        for i in range(len(ranked_u)):
            if ranked_u[i][0] in vocab.vocab_items[u].item_set and vocab.vocab_items[u].item_set[ranked_u[i][0]] > 8:
                if i < 10:
                    dcg += np.log(2) / np.log(i + 2)
                if i < 20:
                    dcg20 += np.log(2) / np.log(i + 2)
                hit += 1
            else:
                num_correct_pairs += hit
        if n_test == 0:
            continue
        cumu_auc += (float)(num_correct_pairs) / ((n_candidate - n_test) * n_test)
        cumu_ndcg_10 += dcg / idcg
        cumu_ndcg_20 += dcg20 / idcg20
        num_t_u += 1
    return (cumu_auc / num_t_u, cumu_ndcg_10 / num_t_u, cumu_ndcg_20 / num_t_u)


def tr_error(vocab, syn0, syn_user):
    test_case = 0
    p5 = 0.0
    r5 = 0.0
    p10 = 0.0
    r10 = 0.0
    for user in range(vocab.user_count):
        #       print user
        test_case += 1
        u_r = 0
        raw_rating = {}
        for item in vocab.vocab_items[user].item_set:
            if vocab.vocab_items[user].item_set[item] > 8:
                u_r += 1
        if u_r == 0:
            continue
        for item1 in range(vocab.item_count):
            if item1 in vocab.vocab_items[user].item_set:
                if vocab.vocab_items[user].item_set[item1] < 8:
                    continue
            pred = 0
            #            print 'item',item1
            for i in range(len(syn0[item1])):
                pred += syn0[item1][i] * syn_user[user][i]
            raw_rating[item1] = pred
        #           print 'pred',pred
        ranked_d = OrderedDict(sorted(raw_rating.items(), key=lambda x: x[1]))
        top10 = ranked_d.keys()[::-1][:10]
        top5 = ranked_d.keys()[::-1][:5]
        g5 = 0.0
        g10 = 0.0
        for i in top5:
            if i in vocab.vocab_items[user].item_set:
                g5 += 1
        for i in top10:
            if i in vocab.vocab_items[user].item_set:
                g10 += 1
        p5 += g5 / 5
        p10 += g10 / 10
        r5 += g5 / u_r
        r10 += g10 / u_r
    return p5 / test_case, p10 / test_case, r5 / test_case, r10 / test_case


def train(fi, pair, comb, split, fo, cbow, neg, dim, alpha, beta, win, min_count, num_processes, binary,
          num_non_neighbors, neighbor_threshold):
    # Read train file to init vocab
    vocab = Vocab(fi, pair, comb, min_count, split)
    user_size = vocab.user_count

    # Init net
    syn0, syn_user, syn1, syn_t = init_net(dim, vocab.item_count, user_size)
    #    print 'Item latent representation ', np.shape(syn0)
    #    print 'User latent representation ', np.shape(syn_user)
    global_word_count = multiprocessing.Value('i', 0)
    last_error = multiprocessing.Value('f', 99999)
    current_error = multiprocessing.Value('f', 9999.0)
    table = None
    if neg > 0:

        print('Initializing unigram table')

        table = UnigramTable(vocab.vocab_4table)
    else:

        print('Initializing Huffman tree')
        vocab.encode_huffman()

    print('Begin Training')
    # Begin training using num_processes workers
    t0 = time.time()
    pool = multiprocessing.Pool(processes=num_processes, initializer=__init_process,
                                initargs=(vocab, syn0, syn_user, syn1, syn_t, table, cbow, neg, dim, alpha, beta,
                                          win, num_processes, global_word_count, last_error, current_error,
                                          num_non_neighbors, neighbor_threshold, fi))
    #    print 'Global',global_word_count.value

    pool.map(train_process, range(num_processes))
    t1 = time.time()

    print('Completed training. Training took', (t1 - t0) / 60, 'minutes')
    print('Start Evaluation\n')
    t2 = time.time()

    result = multiprocessing.Manager().dict()
    pool = multiprocessing.Pool(processes=num_processes, initializer=__init_evaluation,
                                initargs=(vocab, result, syn0, syn_user, syn_t, num_processes))
    result1 = pool.map(predict_parallel, range(num_processes))
    p5, p10, r5, r10 = 0.0, 0.0, 0.0, 0.0
    # print len(result.keys())
    e_time = time.time() - t2
    print('complete evalution, cost ', str(e_time / 60))
    for u in result.keys():
        a = result[u]
        p5 += float(a[0])
        p10 += float(a[1])
        r5 += float(a[2])
        r10 += float(a[3])
    print('#test: %d, p@5: %f, p@10: %f, r@5: %f, r@10: %f\n' % (
        len(result), p5 / len(result), p10 / len(result), r5 / len(result), r10 / len(result)))
    ofile = open('./data/result.txt', 'w')
    ofile.write('p@5:' + str(p5 / len(result)) + '\n')
    ofile.write('r@5:' + str(r5 / len(result)) + '\n')
    ofile.write('p@10:' + str(p10 / len(result)) + '\n')
    ofile.write('r@10:' + str(r10 / len(result)) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-train', help='Training file', dest='fi', required=True)
    path = './data/Gowalla_totalCheckins.txt'
    outpath = './data/out.txt'
    # path = './data/mini_checkins.txt'
    parser.add_argument('-train', help='Training file', dest='fi', default=path)
    parser.add_argument('-pair', help='Pairwise Ranking file', dest='pair')
    parser.add_argument('-comb', help='Combination file', dest='comb')
    parser.add_argument('-split', help='Split for testing', dest='split', type=float, default=0.8)
    parser.add_argument('-model', help='Output model file', dest='fo', default=outpath)
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

    train(args.fi, args.pair, args.comb, args.split, args.fo, bool(args.cbow), args.neg, args.dim, args.alpha,
          args.beta, args.win, args.min_count, args.num_processes, bool(args.binary), args.num_non_neighbors,
          args.neighbor_threshold)
