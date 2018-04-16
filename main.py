from models.TextCNN import model as TextCNN
from models.TextRNN import model as TextRNN
from models.CRNN import model as CRNN
from models.RCNN import model as RCNN
from models.HAN import model as HAN
from models.Ensembles.bagging import model as bagging
from models.Ensembles.stacking import model as stacking

def main():

    # model=TextCNN()
    # model=TextRNN()
    # model=CRNN()
    model=RCNN()
    # model=HAN()
    model.train()
    # for i in range(40,110,10):
    #     model.test(load_path="checkpoints/RCNN/train/model.ckpt-"+str(i))

    # model=bagging()
    # model.train()
    # model.test()

    # model=stacking()
    # model.train_1()
    # model.train_2()
    # model.test()



if __name__=="__main__":
    main()