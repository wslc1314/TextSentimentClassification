import os, logging
from gensim.models import word2vec
import shutil


"""
生成用于训练词向量的语料库。
"""
def getTotalSentences(fileList=("training_label_new.txt",
                                "training_nolabel_new.txt",
                                "testing_data_new.txt")):
    sentences=[]
    for file in fileList:
        with open(os.path.join("dataset", file),'r') as f:
            for line in f.readlines():
                sentences.append(line.strip().split("+++$+++")[1])
    with open("dataset/corpus",'w') as f:
        f.write("\n".join(sentences))


"""
基于word2vec，训练不同维度的词向量。
"""
class MySentences(object):
    def __init__(self, corpus="dataset/corpus"):
        self.corpus=corpus

    def __iter__(self):
        for line in open(self.corpus,'r'):
            yield line.strip().split()


def word2vector(embedding_size=50,window_size=5,training_epochs=5,
                initial_lr=0.025,min_lr=0.0001):
    """
    generate word vectors
    """
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = word2vec.Word2Vec(MySentences(),
                              size=embedding_size,window=window_size,iter=training_epochs,
                              alpha=initial_lr,min_alpha=min_lr,
                              sg=1, min_count=2, workers=4, hs=0, negative=10)
    model_path=os.path.join("word2vec", "model-" + str(embedding_size))
    model.save(model_path)


"""
将用GloVe训练生成的词向量转成能用gensim打开的形式。
"""

# 用gensim打开glove词向量需要在向量的开头增加一行：所有的单词数 词向量的维度。

# 计算行数，就是单词数。
def getFileLineNums(filename):
    f = open(filename, 'r')
    count = 0
    for _ in f:
        count += 1
    return count


# 打开词向量文件，在开始增加一行。
def prepend_line(infile, outfile, line):
    with open(infile, 'r') as old:
        with open(outfile, 'w') as new:
            new.write(str(line) + "\n")
            shutil.copyfileobj(old, new)


def processGloVe(filename):
    num_lines = getFileLineNums(filename)
    filename_=filename.replace(".txt","")
    size=int(filename_.split("-")[-1])
    filename_ = filename_.replace("vectors", "model")
    first_line = "{} {}".format(num_lines, size)
    prepend_line(filename, filename_, first_line)



if __name__=="__main__":

    # getTotalSentences()

    epochs=15
    for size in [25,50,100,200,300]:
        word2vector(embedding_size=size,training_epochs=epochs)
        epochs+=1
        
    # for size in [25,50,100,200,300]:
    #     filename="glove/vectors-"+str(size)+".txt"
    #     processGloVe(filename)