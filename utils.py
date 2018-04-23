import pandas as pd
import numpy as np
import os
from configs import general_config
from data_helpers.utils import readNewFile,loadDict
import logging
import tensorflow as tf

def get_num_params():
    # for v in tf.trainable_variables():
    #     print(v.name)
    #     print(np.prod(v.get_shape().as_list()))
    return np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()
                   if "embedding_matrix" not in v.name.split(":")[0].split("/")
                   and "embedding_matrix_" not in v.name.split(":")[0].split("/")])


class PaddedDataIterator(object):

    def __init__(self, loadPath,vocab2intPath,sent_len_cut=None):
        indices, sentences, labels = readNewFile(file=loadPath, vocab2intPath=vocab2intPath)
        num_words = [len(sentence) for sentence in sentences]
        if isinstance(sent_len_cut, int):
            num_words_=[min(len(sentence),sent_len_cut) for sentence in sentences]
        else:
            num_words_=num_words[:]
        self.df = pd.DataFrame({"id": indices, "sentence": sentences, "label": labels,
                           "sentence_length": num_words,"sentence_length_":num_words_})
        self.total_size=len(self.df)
        self.cursor=0
        self.loop=0
        self.max_len=general_config.max_seq_len
        self.shuffle()

    def shuffle(self):
        self.df=self.df.sample(frac=1).reset_index(drop=True)
        self.cursor=0

    def next(self,batch_size,need_all=False):
        if need_all: # 完整遍历所有数据一轮，常test时用。
            if self.cursor>=self.total_size:
                self.shuffle()
                self.loop+=1
            else:
                batch_size = min(batch_size, self.total_size - self.cursor)
        else:
            if self.cursor+batch_size>self.total_size:
                self.shuffle()
                self.loop += 1
        res=self.df.ix[self.cursor:self.cursor+batch_size-1,:]
        self.cursor+=batch_size
        res_=np.zeros(shape=[batch_size,self.max_len],dtype=np.int32)
        for idx,res_r in enumerate(res_):
            # 少的pad，多的cut。
            tmp_len=min(self.max_len,res["sentence_length"].values[idx])
            res_r[:tmp_len]=res["sentence"].values[idx][:tmp_len]
        return res["id"].values,res_,res["label"].values,res["sentence_length_"].values


class BucketedDataIterator(object):

    def __init__(self, loadPath,vocab2intPath,num_buckets=5):
        indices, sentences, labels = readNewFile(file=loadPath, vocab2intPath=vocab2intPath)
        num_words = [len(sentence) for sentence in sentences]
        self.df = pd.DataFrame({"id": indices, "sentence": sentences, "label": labels,
                           "sentence_length": num_words})
        df=self.df.sort_values("sentence_length").reset_index(drop=True)
        self.total_size=len(df)
        part_size=self.total_size//num_buckets
        self.dfs=[]
        for i in range(num_buckets):
            self.dfs.append(df.ix[i*part_size:(i+1)*part_size-1])
        self.dfs[num_buckets-1].append(df.ix[num_buckets*part_size:self.total_size-1])
        self.num_buckets=num_buckets
        self.cursor=np.array([0]*num_buckets)
        self.p_list=[1/self.num_buckets]*self.num_buckets
        self.loop=0
        self.max_len=general_config.max_seq_len
        self.shuffle()

    def shuffle(self):
        for i in range(self.num_buckets):
            self.dfs[i]=self.dfs[i].sample(frac=1).reset_index(drop=True)
            self.cursor[i]=0

    def next(self,batch_size,need_all=False):
        for i in range(self.num_buckets):
            if need_all:
                if self.cursor[i]>=len(self.dfs[i]):
                    self.p_list[i]=0
            else:
                if self.cursor[i]+batch_size>len(self.dfs[i]):
                    self.p_list[i] = 0
        if sum(self.p_list) == 0:
            self.shuffle()
            self.loop += 1
            self.p_list = [1 / self.num_buckets] * self.num_buckets
        else:
            times = 1 / sum(self.p_list)
            self.p_list = [times * p for p in self.p_list]
        selected=np.random.choice(a=np.arange(self.num_buckets),size=1,p=self.p_list)[0]
        if need_all:
            batch_size=min(batch_size,len(self.dfs[selected])-self.cursor[selected])
        res=self.dfs[selected].ix[self.cursor[selected]:self.cursor[selected]+batch_size-1,:]
        self.cursor[selected]+=batch_size
        tmp_max_len=max(res["sentence_length"].values)
        max_len=min(tmp_max_len,self.max_len)
        res_=np.zeros(shape=[batch_size,max_len],dtype=np.int32)
        for idx,res_r in enumerate(res_):
            # 少的pad，多的cut。
            tmp_len=min(max_len,res["sentence_length"].values[idx])
            res_r[:tmp_len]=res["sentence"].values[idx][:tmp_len]
        return res["id"].values,res_,res["label"].values,res["sentence_length"].values


def ensure_dir_exist(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir


def WriteToSubmission(res,fileName):
    fileDir=os.path.dirname(fileName)
    ensure_dir_exist(fileDir)
    tmp=pd.DataFrame(res,columns=["id","label"])
    tmp=tmp.sort_values(by="id",axis=0,ascending=True)
    tmp.to_csv(fileName,index=False)


"""
将单词列表形式的句子转为句子列表形式的文档，
以"."、"?"、"!"为句子分隔符。
"""
def sentence2doc(words,v2i=None):
    if v2i is None:
        selected=[".","?","!"]
    else:
        selected=[v2i["."],v2i["?"],v2i["!"]]
    doc=[]
    sentence=[]
    for word in words:
        sentence.append(word)
        if word in selected:
            doc.append(sentence)
            sentence=[]
    if len(sentence)>0:
        doc.append(sentence)
    if len(doc)==0:
        print(words)
    return doc


class BucketedDataIteratorForDoc(object):
    def __init__(self, loadPath,vocab2intPath,num_buckets=5):
        indices, sentences, labels = readNewFile(file=loadPath, vocab2intPath=vocab2intPath)
        v2i=loadDict(vocab2intPath)
        docs=[]
        num_sentences=[]
        num_words=[]
        num_words_flat=[]
        for sentence in sentences:
            doc=sentence2doc(sentence,v2i)
            docs.append(doc)
            num_sentences.append(len(doc))
            num_words_=[len(_) for _ in doc]
            num_words.append(num_words_)
            num_words_flat.extend(num_words_)
        # print(max(num_sentences))
        # print(max(num_words_flat))
        # print(num_words[:5])
        self.df = pd.DataFrame({"id": indices, "doc":docs, "label": labels,
                           "doc_length": num_sentences,"sentence_length":num_words})
        df=self.df.sort_values("doc_length").reset_index(drop=True)
        self.total_size=len(df)
        part_size=self.total_size//num_buckets
        self.dfs=[]
        for i in range(num_buckets):
            self.dfs.append(df.ix[i*part_size:(i+1)*part_size-1])
        self.dfs[num_buckets-1].append(df.ix[num_buckets*part_size:self.total_size-1])
        self.num_buckets=num_buckets
        self.cursor=np.array([0]*num_buckets)
        self.p_list=[1/self.num_buckets]*self.num_buckets
        self.loop=0
        self.shuffle()

    def shuffle(self):
        for i in range(self.num_buckets):
            self.dfs[i]=self.dfs[i].sample(frac=1).reset_index(drop=True)
            self.cursor[i]=0

    def next(self,batch_size,need_all=False):
        for i in range(self.num_buckets):
            if need_all:
                if self.cursor[i]>=len(self.dfs[i]):
                    self.p_list[i]=0
            else:
                if self.cursor[i]+batch_size>len(self.dfs[i]):
                    self.p_list[i] = 0
        if sum(self.p_list) == 0:
            self.shuffle()
            self.loop += 1
            self.p_list = [1 / self.num_buckets] * self.num_buckets
        else:
            times = 1 / sum(self.p_list)
            self.p_list = [times * p for p in self.p_list]
        selected=np.random.choice(a=np.arange(self.num_buckets),size=1,p=self.p_list)[0]
        if need_all:
            batch_size=min(batch_size,len(self.dfs[selected])-self.cursor[selected])
        res=self.dfs[selected].ix[self.cursor[selected]:self.cursor[selected]+batch_size-1,:]
        self.cursor[selected]+=batch_size
        max_doc_len=np.max(res["doc_length"].values)
        sentence_length_flat=[]
        for l in res["sentence_length"].values:
            sentence_length_flat.extend(l)
        max_sen_len=np.max(sentence_length_flat)
        res_=np.zeros(shape=[batch_size,max_doc_len,max_sen_len],dtype=np.int32)
        res_sen_len=np.zeros(shape=[batch_size,max_doc_len],dtype=np.int32)
        for b in range(batch_size):
            doc_len=res["doc_length"].values[b]
            for d in range(doc_len):
                sen_len=res["sentence_length"].values[b][d]
                # 少的pad。
                res_[b,d,:sen_len]=res["doc"].values[b][d]
                res_sen_len[b,d]=res["sentence_length"].values[b][d]
        # res_=np.reshape(res_,newshape=(batch_size,-1))
        # print(res_.shape)
        # print(res_sen_len.shape)
        return res["id"].values,res_,res["label"].values,res["doc_length"].values,res_sen_len


def my_logger(logging_path):
    # 生成日志
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    logger.handlers = []
    assert len(logger.handlers) == 0
    handler = logging.FileHandler(logging_path)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # console.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger