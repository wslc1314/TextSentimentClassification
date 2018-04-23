from models.Others.LR import model as LR
from models.Others.NB import model as NB
from models.TextCNN import model as TextCNN
from models.TextRNN import model as TextRNN
from models.CRNN import model as CRNN
from models.RCNN import model as RCNN
from models.HAN import model as HAN
from models.Ensembles.bagging import model as bagging
from models.Ensembles.stacking import model as stacking
from configs import general_config


def main():

    # # model=LR()
    # model=NB()
    # model.train()
    # model.test()

    # # for model_type in ["baseline","static","nonstatic","multichannel"]:
    # #     model=TextCNN(model_type=model_type)
    # #     model.fit(with_validation=True)
    # # for model_type in ["baseline", "static", "nonstatic", "multichannel"]:
    # #     model = TextCNN(model_type=model_type)
    # #     model.evaluate(load_path="checkpoints/TextCNN/"+model_type+"/train_valid",
    # #                    validFile=general_config.train_file,
    # #                    vocab2intPath=general_config.local_nonstatic_v2i_path)
    # #     model.evaluate(load_path="checkpoints/TextCNN/"+model_type+"/train_valid",
    # #                    validFile=general_config.valid_file,
    # #                    vocab2intPath=general_config.local_nonstatic_v2i_path)
    # model=TextCNN(model_type="nonstatic")
    # # model.fit(with_validation=False,num_epochs=130,num_visual=0)
    # model.evaluate(load_path="checkpoints/TextCNN/nonstatic/train")
    # model.predict(load_path="checkpoints/TextCNN/nonstatic/train")
    #
    # model=TextRNN()
    # # # model.fit(with_validation=True)
    # # model.evaluate(load_path="checkpoints/TextRNN/train_valid",
    # #                    validFile=general_config.train_file,
    # #                    vocab2intPath=general_config.local_nonstatic_v2i_path)
    # # model.evaluate(load_path="checkpoints/TextRNN/train_valid",
    # #                validFile=general_config.valid_file,
    # #                vocab2intPath=general_config.local_nonstatic_v2i_path)
    # # model.fit(with_validation=False, num_epochs=150, num_visual=0)
    # model.evaluate(load_path="checkpoints/TextRNN/train")
    # model.predict(load_path="checkpoints/TextRNN/train")
    #
    # model=CRNN()
    # # # model.fit(with_validation=True)
    # # model.evaluate(load_path="checkpoints/CRNN/train_valid",
    # #                    validFile=general_config.train_file,
    # #                    vocab2intPath=general_config.local_nonstatic_v2i_path)
    # # model.evaluate(load_path="checkpoints/CRNN/train_valid",
    # #                validFile=general_config.valid_file,
    # #                vocab2intPath=general_config.local_nonstatic_v2i_path)
    # # model.fit(with_validation=False, num_epochs=70, num_visual=0)
    # model.evaluate(load_path="checkpoints/CRNN/train")
    # model.predict(load_path="checkpoints/CRNN/train")
    #
    # model=RCNN()
    # # # model.fit(with_validation=True)
    # # model.evaluate(load_path="checkpoints/RCNN/train_valid",
    # #                    validFile=general_config.train_file,
    # #                    vocab2intPath=general_config.local_nonstatic_v2i_path)
    # # model.evaluate(load_path="checkpoints/RCNN/train_valid",
    # #                validFile=general_config.valid_file,
    # #                vocab2intPath=general_config.local_nonstatic_v2i_path)
    # # model.fit(with_validation=False, num_epochs=50, num_visual=0)
    # model.evaluate(load_path="checkpoints/RCNN/train")
    # model.predict(load_path="checkpoints/RCNN/train")
    #
    # model=HAN()
    # # # model.fit(with_validation=True)
    # # model.evaluate(load_path="checkpoints/HAN/train_valid",
    # #                    validFile=general_config.train_file,
    # #                    vocab2intPath=general_config.local_nonstatic_v2i_path)
    # # model.evaluate(load_path="checkpoints/HAN/train_valid",
    # #                validFile=general_config.valid_file,
    # #                vocab2intPath=general_config.local_nonstatic_v2i_path)
    # # model.fit(with_validation=False, num_epochs=110, num_visual=0)
    # model.evaluate(load_path="checkpoints/HAN/train")
    # model.predict(load_path="checkpoints/HAN/train")

    # model=bagging()
    # # model.fit()
    # model.evaluate()
    # model.predict()

    model=stacking()
    model.train_1()
    model.train_2()
    # model.evaluate()
    # model.predict()



if __name__=="__main__":
    main()