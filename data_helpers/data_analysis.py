import matplotlib.pyplot as plt


"""
1、分析训练数据中标签分布情况；
2、分析训练数据、测试数据中总句子数、句子中单词数以及最大句子中单词数。
"""


def label_distribution(trainFile):
    labels=[]
    with open(trainFile,'r') as f:
        raw_data=f.readlines()
        for line in raw_data:
            line=line.strip().split("+++$+++")
            label=int(line[2])
            assert label in [0,1],"Invalid label value!"
            labels.append(label)
    neg_count=labels.count(0)
    pos_count=len(labels)-neg_count
    counts=[neg_count,pos_count]
    labels=["negative","positive"]
    fig=plt.figure(figsize=(9,9))
    # 画饼图（数据，数据对应的标签，百分数保留两位小数点）
    plt.pie(counts, labels=labels, autopct='%1.2f%%')
    plt.title('Train Label Distribution', bbox={'facecolor': '0.8', 'pad': 5})
    plt.show()
    savePath=trainFile.split(".")[0]+"_ld.png"
    fig.savefig(savePath)
    plt.close()


def sentences_attributes(sentencesFile):
    """
    get num_sentences,num_words,max_num_words from sentencesFile.
    """
    sen_len=[]
    with open(sentencesFile,'r') as f:
        raw_data=f.readlines()
        num_sentences=len(raw_data)
        for line in raw_data:
            line=line.strip().split("+++$+++")
            sentence=line[1].split()
            sen_len.append(len(sentence))
    print(num_sentences)
    print(sen_len[:5])
    max_sen_len=max(sen_len)
    fig=plt.figure(figsize=(16,9))
    plt.hist(sen_len,bins=20)
    plt.title(sentencesFile.split("/")[-1].replace(".txt",""),bbox={'facecolor': '0.8', 'pad': 5})
    plt.show()
    fig.savefig(sentencesFile.replace(".txt",".png"))
    plt.close()
    print(max_sen_len)
    return num_sentences, sen_len, max_sen_len



if __name__=="__main__":
    label_distribution(trainFile="dataset/training_label_new.txt")
    _,_,max_train_len=sentences_attributes(sentencesFile="dataset/training_label_new.txt")
    _,_,max_test_len=sentences_attributes(sentencesFile="dataset/testing_data_new.txt")
    print(max(max_train_len,max_test_len)) # 39