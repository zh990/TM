import codecs
from gensim import corpora
from gensim.models import ldaseqmodel
import time
from gensim.test.utils import datapath
import os

#
# train = []
# fp = codecs.open('D:/pycharm/02_LDAexercise/test2f.txt', 'r', encoding='utf-8')
# for line in fp:
#     if line != '':
#         line = line.split()
#         train.append([w for w in line])
#
# # 生成DTM主题模型
# time0 = time.time()
# dictionary = corpora.Dictionary(train)
# # dictionary.filter_extremes(no_below=3, no_above=0.9)
# corpus0 = [dictionary.doc2bow(text) for text in train[0:(len(train)//3)]]
# corpus1 = [dictionary.doc2bow(text) for text in train[len(train)//3:(len(train)*2//3)]]
# corpus2 = [dictionary.doc2bow(text) for text in train[(len(train)*2//3):len(train)]]
# corpus = [sum(corpus0, [])] + [sum(corpus1, [])] + [sum(corpus2, [])]
# time1 = time.time()
# print("build corpus spends: ", time1 - time0)
# K = 7
# model = ldaseqmodel.LdaSeqModel(corpus=corpus, id2word=dictionary,
#                                 time_slice=[1, 1, 1],
#                                 num_topics=K)
# print("build model spends: ", time.time() - time1)

temp_file = datapath("dtm_model")  # 虽不是具体路径，但是可以保存模型并直接调用哦！
# model.save(temp_file)

# Load a potentially pre-trained model from disk.
model = ldaseqmodel.LdaSeqModel.load(temp_file)
for i in range(7):
    print("第%d 个主题下每个时间片的主题词概率：" % i, model.print_topic_times(i), '\n')   # 获取每个时间片下第0个主题的主题词概率