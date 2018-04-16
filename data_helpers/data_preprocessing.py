from string import punctuation
import re
from sklearn.model_selection import train_test_split


"""
1、对原始数据进行清洗后进行保存，
2、将训练数据分为训练集和验证集。
"""


def process_word(word):
    """
    对全英文单词，保证字母重复的最大次数不超过2次；
    对含标点符号的单词，取其第一个字符。
    """
    if word[0] in punctuation:
        return word[0]
    if re.match(pattern="[a-zA-Z]+",string=word) is not None:
        count = 1
        start = 0
        end = 0
        while (start < len(word) - 1):
            if end < len(word) - 1 and word[end + 1] == word[end]:
                count += 1
                end += 1
            else:
                if count > 2:
                    word = word[:start] + word[end:]
                    if end == len(word) - 1:
                        return word
                start += 1
                end = start
                count = 1
    return word


def process_str_sentence(str_sentence):
    sentence = str_sentence.lower().strip().split()
    # print(sentence)
    sentence = [process_word(word) for word in sentence]
    # print(sentence)
    return sentence


def processOriginalData(loadFile,is_train=True,with_label=True):
    indices = []
    sentences = []
    labels = []
    loadName=loadFile.split('.')[0]
    saveFile=loadName+"_new"+".txt"
    with open(loadFile,'r') as f:
        if is_train:
            raw_data = f.readlines()
            if with_label:
                # reading training_label
                for i in range(len(raw_data)):
                    line=raw_data[i]
                    line = line.strip().split("+++$+++")
                    sentence = process_str_sentence(line[1])
                    if len(sentence) > 0:
                        print(i)
                        sentences.append(sentence)
                        indices.append(str(i))
                        labels.append(line[0])
                with open(saveFile, 'w') as f:
                    for i in range(len(sentences)):
                        sentence=" ".join(sentences[i])
                        s=" +++$+++ ".join([indices[i],sentence,labels[i]])
                        f.write(s+"\n")
            else:
                # reading training_nolabel
                for i in range(len(raw_data)):
                    line=raw_data[i].strip()
                    sentence = process_str_sentence(line)
                    if len(sentence) > 0:
                        print(i)
                        sentences.append(sentence)
                        indices.append(str(i))
                with open(saveFile, 'w') as f:
                    for i in range(len(sentences)):
                        sentence=" ".join(sentences[i])
                        s=" +++$+++ ".join([indices[i],sentence])
                        f.write(s + "\n")
        else:
            # reading testing_data
            raw_data=f.readlines()[1:]
            for line in raw_data:
                line=line.strip().split(',')
                id=line[0]
                line=" ".join(line[1:])
                sentence = process_str_sentence(line)
                if len(sentence) > 0:
                    print(int(id))
                    sentences.append(sentence)
                    indices.append(id)
            with open(saveFile, 'w') as f:
                for i in range(len(sentences)):
                    sentence=" ".join(sentences[i])
                    s=" +++$+++ ".join([indices[i],sentence])
                    f.write(s + "\n")


def split_train_val(trainFile, validation_size=0.1):
    """
    split train into train and valid, then save them
    """
    with open(trainFile, 'r') as f:
        raw_data = f.readlines()
    train, valid = train_test_split(raw_data, test_size=validation_size)
    with open("dataset/train.txt", 'w') as f:
        f.writelines(train)
    with open("dataset/valid.txt", 'w') as f:
        f.writelines(valid)



if __name__=="__main__":

    # processOriginalData(loadFile="dataset/training_label.txt",is_train=True,with_label=True)
    # processOriginalData(loadFile="dataset/training_nolabel.txt",is_train=True,with_label=False)
    # processOriginalData(loadFile="dataset/testing_data.txt",is_train=False,with_label=False)

    split_train_val(trainFile="dataset/training_label_new.txt")