import json
import os
from gensim import models
import numpy as np
import pandas as pd


"""
预训练的词向量使用方法：
1、非静态
1）出现在预训练的词向量中的词，也出现在训练集中：
   直接使用词向量初始化其embeddings；
2）未出现在预训练的词向量中的词却出现在训练集中：
   选取其中出现频次较高的词进行词向量随机初始化，其余做<unk>处理，使它们与预训练词向量尽量同分布。
   此处因为使用所有给定的文本进行词向量的预训练，所以训练集中只有出现1次的词在词向量中无对应向量，将它们直接做<unk>处理；
3）出现在预训练词向量中的词，未出现在训练集中，却出现在测试集中：
   若在嵌入词向量之前使用全连接层，那就使用其对应词向量，否则，当做<unk>处理；
4）只出现在测试集中的词，使用<unk>处理。
2、静态
1）出现在预训练的词向量中的词，只要出现在训练集或测试集中：
   直接使用词向量初始化其embeddings；
2）未出现在预训练的词向量中的词却出现在训练集中：
   选取其中出现频次较高的词进行词向量随机初始化，其余做<unk>处理，使它们与预训练词向量尽量同分布。
   此处因为使用所有给定的文本进行词向量的预训练，所以训练集中只有出现1次的词在词向量中无对应向量，将它们直接做<unk>处理；
3）只出现在测试集中的词，使用<unk>处理。
"""


def saveDict(dicts,saveFile):
    with open(saveFile,'w') as f:
        json.dump(dicts,f)


def loadDict(loadFile):
    with open(loadFile,'r') as f:
        dicts=json.load(f)
    return dicts


def readNewFile(file,vocab2intPath=None):
    indices,sentences,labels=[],[],[]
    if isinstance(vocab2intPath,str):
        vocab2int=loadDict(vocab2intPath)
    with open(file,'r') as f:
        raw_data=f.readlines()
        for line in raw_data:
            line=line.strip().split("+++$+++")
            indices.append(int(line[0]))
            if isinstance(vocab2intPath,str):
                sentences.append([vocab2int.get(word, vocab2int["<unk>"])for word in line[1].split()])
            else:
                sentences.append(line[1].split())
            if len(line) > 2:
                labels.append(int(line[2]))
            else:
                labels.append([])
        return indices, sentences, labels


"""
记录全局静态/动态词-序号、序号-词对应关系。
"""
def createGlobalWordDict(wv_path,is_static):
    trainFile = "dataset/training_label_new.txt"
    _,sentences,_=readNewFile(trainFile)
    savePath=os.path.join(os.path.dirname(trainFile),os.path.basename(trainFile).replace(".txt","").split('_')[0])
    if is_static:
        testFile="dataset/testing_data_new.txt"
        _, sentences_, _ = readNewFile(testFile)
        sentences.extend(sentences_)
        savePath += "_testing"
    words={}
    for sentence in sentences:
        for word in sentence:
            try:
                words[word]+=1
            except:
                words[word]=1
    wv_dir=os.path.dirname(wv_path)
    if wv_dir=="glove":
        # 用GloVe训练的词向量中自带<unk>。
        model = models.KeyedVectors.load_word2vec_format(wv_path, binary=False)
    else:
        model=models.Word2Vec.load(wv_path)
        # 取出现频次为2的所有词向量的均值作为<unk>初始值。
        if not os.path.exists(wv_dir+"/infrequent.json"):
            infrequentWords={}
            infrequentWords[2]=[]
            for (word, vocab) in model.wv.vocab.items():
                if vocab.count==2:
                    infrequentWords[2].append(word)
            print(len(infrequentWords[2])) # 48414
            saveDict(infrequentWords,wv_dir+"/infrequent.json")
    wv_vocab=model.wv.index2word
    print("Size for text words: ",len(words.keys())) # 115751/77046
    print("Size for wv words: ",len(wv_vocab)) # 126438-glove,126437-word2vec
    vocab=list(set(wv_vocab)&set(words.keys()))
    print("Size for their intersection: ",len(vocab)) # 104991/71667
    int_to_vocab = dict(enumerate(['<pad>']+ vocab+['<unk>']))
    vocab_to_int = dict(zip(int_to_vocab.values(), int_to_vocab.keys()))
    print(vocab_to_int['<pad>']) # 0
    print(vocab_to_int['<unk>']) # 104992/71668
    print(len(list(int_to_vocab.keys()))) # 104993/71669
    saveDict(int_to_vocab,
             savePath+"_i2v.json")
    saveDict(vocab_to_int,
             savePath+"_v2i.json" )


"""
根据训练数据的不同生成不同的动态词-序号、序号-词字典。
"""
def getNonstaticWordDict(trainFile,global_v2i_path="dataset/training_v2i.json"):
    global_v2i=loadDict(global_v2i_path)
    words={}
    _,sentences,_=readNewFile(trainFile)
    savePath=os.path.join(os.path.dirname(trainFile),os.path.basename(trainFile).replace(".txt","").split('_')[0])
    for sentence in sentences:
        for word in sentence:
            try:
                words[word]+=1
            except:
                words[word]=1
    print("Size for text words: ",len(words.keys())) # 72450
    print("Size for global words: ",len(global_v2i.keys())) # 71669
    vocab=list(set(global_v2i.keys())&set(words.keys()))
    print("Size for their intersection: ",len(vocab)) # 67588
    vocab =['<pad>']+ vocab+['<unk>']
    v2i={}
    i2v={}
    for word in vocab:
        id=global_v2i[word]
        v2i[word]=id
        i2v[id]=word
    print(v2i['<pad>']) # 0
    print(v2i['<unk>']) # 71668
    print(len(v2i.keys())) # 67590
    saveDict(i2v,savePath+"_i2v.json")
    saveDict(v2i,savePath+"_v2i.json" )


def load_embedding_matrix(wv_path,int2vocabPath="dataset/training_i2v.json"):
    int2vocab=loadDict(int2vocabPath)
    vocab2int=loadDict(int2vocabPath.replace("i2v","v2i"))
    vocab_size=vocab2int["<unk>"]+1
    assert vocab_size==len(int2vocab.keys()),"Here must be a global dict, no matter static or nonstatic!"
    embedding_size=int(wv_path.split("-")[-1])
    embeddings = np.random.uniform(low=-0.05,high=0.05,size=(vocab_size, embedding_size))
    if "glove" in wv_path.split("/"):
        model = models.KeyedVectors.load_word2vec_format(wv_path, binary=False)
        embeddings[vocab_size - 1] = model['<unk>']
    else:
        model = models.Word2Vec.load(wv_path)
        infrequentWords = loadDict(os.path.dirname(wv_path)+"/infrequent.json")
        tmp = np.zeros([embedding_size, ])
        for w in infrequentWords[str(2)]:
            tmp += model[w]
        embeddings[vocab_size - 1] = tmp / len(infrequentWords[str(2)])
    for i in range(1,vocab_size-1):
        word=int2vocab[str(i)]
        embeddings[i] = model[word]
    return embeddings


def create_visual_metadata(int2vocab_path):
    int_to_vocab = loadDict(int2vocab_path)
    labels = [(int_to_vocab[str(i)], i) for i in int_to_vocab.keys()]
    savePath=int2vocab_path.replace("i2v.json","metadata.tsv")
    pd.DataFrame(labels, columns=['word', 'idx']).to_csv(savePath, index=False,sep='\t')



if __name__=="__main__":

    createGlobalWordDict(wv_path="glove/model-25",is_static=True)
    createGlobalWordDict(wv_path="word2vec/model-25",is_static=False)

    getNonstaticWordDict(trainFile="dataset/train.txt",global_v2i_path="dataset/training_v2i.json")

    load_embedding_matrix(wv_path="glove/model-25",int2vocabPath="dataset/training_i2v.json")
    load_embedding_matrix(wv_path="word2vec/model-25",int2vocabPath="dataset/training_testing_i2v.json")

    create_visual_metadata(int2vocab_path="dataset/train_i2v.json")
    create_visual_metadata(int2vocab_path="dataset/training_i2v.json")
    create_visual_metadata(int2vocab_path="dataset/training_testing_i2v.json")