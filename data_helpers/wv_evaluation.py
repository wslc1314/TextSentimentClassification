from gensim import models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


""""
1、可视化词向量；
2、通过类比关系评价词向量。
"""
def visualizeWordVec(wv_path,saveName):
    file_type=wv_path.split("/")[0]
    if file_type=="glove":
        model = models.KeyedVectors.load_word2vec_format(wv_path, binary=False)
    else:
        model=models.Word2Vec.load(wv_path)
    print("------我是分割线------")
    print(wv_path)
    for i in model.wv.most_similar(positive=["woman","king"], negative=["man"]):
        print(i[0], i[1])

    visualizeWords = [
        "good","better","best",
        "bad","worse","worst",
        "great", "brilliant", "wonderful",
        "boring", "waste", "dumb",
        "the", "a", "an",
        "?", "!"]
    visualizeVecs=[]
    for i in visualizeWords:
        visualizeVecs.append(model[i])
    visualizeVecs = np.array(visualizeVecs).reshape((len(visualizeWords),-1))

    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(visualizeVecs)
    fig=plt.figure(figsize=(16,9))
    plt.plot([Y[0,0],Y[1,0]],[Y[0,1],Y[1,1]],color='r')
    plt.plot([Y[3,0],Y[4,0]],[Y[3,1],Y[4,1]],color='b')

    plt.scatter(Y[:, 0], Y[:, 1])
    for label, x, y in zip(visualizeWords, Y[:, 0], Y[:, 1]):
        plt.text(x, y, label,bbox=dict(facecolor='green', alpha=0.1))
    plt.xlim((np.min(Y[:, 0])-10, np.max(Y[:, 0])+10))
    plt.ylim((np.min(Y[:, 1])-10, np.max(Y[:, 1])+10))
    plt.title(saveName.replace('/','-').replace(".png",""))
    plt.show()
    fig.savefig(saveName)
    plt.close()



if __name__=="__main__":

    # word2vec
    wv_path="word2vec/model-"
    for i in [25,50,100,200,300]:
        wv_path_=wv_path+str(i)
        visualizeWordVec(wv_path_,saveName=wv_path_+".png")

    # glove
    wv_path="glove/model-"
    for i in [25,50,100,200,300]:
        wv_path_=wv_path+str(i)
        visualizeWordVec(wv_path_,saveName=wv_path_+".png")

