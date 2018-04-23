from configs import general_config,bagging_config,modelDict
from models.TextCNN import model as TextCNN
from models.TextRNN import model as TextRNN
from models.CRNN import model as CRNN
from models.RCNN import model as RCNN
from models.HAN import model as HAN
from utils import ensure_dir_exist,WriteToSubmission,my_logger
from data_helpers.utils import getNonstaticWordDict,create_visual_metadata
import numpy as np
import os


def createRandomData(num_random):
    trainingFile = general_config.training_file
    with open(trainingFile, 'r') as f:
        raw_data = np.asarray(f.readlines())
    total_size=len(raw_data)
    saveDir = ensure_dir_exist(general_config.data_dir + "/random")
    for i in range(num_random):
        trainFile = saveDir + "/training" + str(i) + ".txt"
        if os.path.exists(trainFile):
            continue
        np.random.seed(seed=10*i)
        indices=np.random.choice(total_size,total_size,replace=True)
        with open(trainFile,'w') as f:
            f.writelines(raw_data[indices])
        getNonstaticWordDict(trainFile=trainFile,global_v2i_path=general_config.global_nonstatic_v2i_path)
        create_visual_metadata(int2vocab_path=trainFile.replace(".txt","_i2v.json"))


class model(object):

    def __init__(self,base_model_list=bagging_config.base_model_list):
        self.base_model_list = base_model_list.split("-")
        self.num_random=len(self.base_model_list)
        self.dataDir = general_config.data_dir + "/random"
        createRandomData(self.num_random)

        self.models = []
        self.models_name=[]
        for i in range(self.num_random):
            base_model = self.base_model_list[i]
            assert base_model in ["1", "2", "3", "4","5"], "Invalid base model type!"
            if base_model == "1":
                model = TextCNN()
            elif base_model == "2":
                model = TextRNN()
            elif base_model == "3":
                model = CRNN()
            elif base_model=="4":
                model = RCNN()
            else:
                model=HAN()
            self.models.append(model)
            self.models_name.append(modelDict[base_model])
        self.logDir = ensure_dir_exist(general_config.log_dir + "/bagging/" + "-".join(self.models_name))
        self.saveDir = ensure_dir_exist(general_config.save_dir + "/bagging/" + "-".join(self.models_name))
        self.logger=my_logger(self.logDir+"/log.txt")

    def fit(self,num_epochs_list=bagging_config.num_epochs_list):
        num_epochs=[int(i) for i in num_epochs_list.split('-')]
        assert len(num_epochs)==self.num_random
        for i in range(self.num_random):
            model=self.models[i]
            model_name=self.models_name[i]
            num_epoch=num_epochs[i]
            trainFile = self.dataDir + "/training" + str(i) + ".txt"
            log_dir = self.logDir+ "/" + str(i)+"_"+model_name
            save_dir = self.saveDir+ "/" + str(i)+"_"+model_name
            model.fit(trainFile=trainFile,with_validation=False,
                      log_dir=log_dir, save_dir=save_dir,num_epochs=num_epoch,
                      num_visual=0)

    def evaluate(self,load_epochs_list=bagging_config.load_epochs_list,
                 validFile=None):
        if load_epochs_list is None:
            load_epochs=None
        else:
            load_epochs = load_epochs_list.split("-")
            assert len(load_epochs) == self.num_random
        if validFile is None:
            trainFile = general_config.training_file
        else:
            trainFile=validFile
        avg_loss,avg_acc=0.,0.
        for i in range(self.num_random):
            model = self.models[i]
            model_name = self.models_name[i]
            vocab2intPath = self.dataDir + "/training" + str(i) + "_v2i.json"
            if model_name == "TextCNN":
                load_path = self.saveDir + "/" + str(i) + "_" + model_name + "/nonstatic/train"
            else:
                load_path = self.saveDir + "/" + str(i) + "_" + model_name + "/train"
            if load_epochs is not None:
                load_path+="/model.ckpt-"+load_epochs[i]
            loss,acc = model.evaluate(validFile=trainFile, vocab2intPath=vocab2intPath,load_path=load_path)
            avg_loss+=loss
            avg_acc+=acc
        avg_loss/=self.num_random
        avg_acc/=self.num_random
        self.logger.info("Loss: %.4f, Accuracy: %.4f "% (avg_loss, avg_acc))
        return avg_loss,avg_acc

    def predict(self,load_epochs_list=bagging_config.load_epochs_list,
                testFile=None):
        if load_epochs_list is None:
            load_epochs=None
        else:
            load_epochs = load_epochs_list.split("-")
            assert len(load_epochs)==self.num_random
        if testFile is None:
            testFile =general_config.testing_file
        tmp_res={}
        res_dir = ensure_dir_exist(self.saveDir.replace("checkpoints", "results"))
        for i in range(self.num_random):
            model=self.models[i]
            model_name=self.models_name[i]
            vocab2intPath = self.dataDir + "/training" + str(i) + "_v2i.json"
            if model_name == "TextCNN":
                load_path = self.saveDir + "/" + str(i) + "_" + model_name + "/nonstatic/train"
            else:
                load_path = self.saveDir + "/" + str(i) + "_" + model_name + "/train"
            resPath = res_dir + "/" + str(i) + "_predicted.csv"
            if load_epochs is not None:
                load_path+="/model.ckpt-"+load_epochs[i]
                resPath = resPath+"-"+load_epochs[i]
            res=model.predict(testFile=testFile,vocab2intPath=vocab2intPath,
                              load_path=load_path,resPath=resPath)
            for key, value in res.items():
                try:
                    tmp_res[key][value]+=1
                except:
                    tmp = {}
                    for j in range(general_config.num_classes):
                        tmp[j] = 0
                    tmp[value]+=1
                    tmp_res[key]=tmp
        res=[]
        for id,item in tmp_res.items():
            tmp=sorted(item.items(),key=lambda d:d[1],reverse=True)[0][0]
            res.append([id,tmp])
        if load_epochs_list is None:
            WriteToSubmission(res, fileName=res_dir + "/predicted.csv")
        else:
            WriteToSubmission(res,fileName=res_dir+"/predicted.csv-"+load_epochs_list)