from configs import general_config,bagging_config,modelDict
from models.TextCNN import model as TextCNN
from models.TextRNN import model as TextRNN
from models.CRNN import model as CRNN
from models.RCNN import model as RCNN
from utils import ensure_dir_exist,WriteToSubmission
from data_helpers.utils import getNonstaticWordDict,create_visual_metadata
import numpy as np
import os


def createRandomData(num_random):
    trainingFile = general_config.data_dir + "/training_label_new.txt"
    with open(trainingFile, 'r') as f:
        raw_data = np.asarray(f.readlines())
    total_size=len(raw_data)
    saveDir = ensure_dir_exist(general_config.data_dir + "/random")
    for i in range(num_random):
        trainFile = saveDir + "/training" + str(i) + ".txt"
        if os.path.exists(trainFile):
            continue
        np.random.seed(seed=10**i)
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
            assert base_model in ["1", "2", "3", "4"], "Invalid base model type!"
            if base_model == "1":
                model = TextCNN()
            elif base_model == "2":
                model = TextRNN()
            elif base_model == "3":
                model = CRNN()
            else:
                model = RCNN()
            self.models.append(model)
            self.models_name.append(modelDict[base_model])
        self.logDir = ensure_dir_exist(general_config.log_dir + "/bagging/" + "-".join(self.models_name))
        self.saveDir = ensure_dir_exist(general_config.save_dir + "/bagging/" + "-".join(self.models_name))

    def train(self):
        for i in range(self.num_random):
            model=self.models[i]
            model_name=self.models_name[i]
            trainFile = self.dataDir + "/training" + str(i) + ".txt"
            log_dir = self.logDir+ "/" + str(i)+"_"+model_name
            save_dir = self.saveDir+ "/" + str(i)+"_"+model_name
            model.train(trainFile=trainFile,with_validation=False,
                        log_dir=log_dir, save_dir=save_dir)

    def test(self,load_model_epoch_list=bagging_config.load_model_epoch_list):
        load_epoch_list=load_model_epoch_list.split("-")
        assert len(load_epoch_list)==self.num_random,"Invalid load model paths!"
        testFile = os.path.join(general_config.data_dir, "testing_data_new.txt")
        tmp_res={}
        res_dir = ensure_dir_exist(
            (self.saveDir + "/" + load_model_epoch_list).replace("checkpoints", "results"))
        for i in range(self.num_random):
            model=self.models[i]
            model_name=self.models_name[i]
            load_epoch=load_epoch_list[i]
            vocab2intPath = self.dataDir + "/training" + str(i) + "_v2i.json"
            load_path=self.saveDir+"/"+str(i)+"_"+model_name+"/train/model.ckpt-"+load_epoch
            res=model.test(testFile=testFile,vocab2intPath=vocab2intPath,
                       load_path=load_path,resPath=res_dir+"/"+str(i)+"_predicted_"+load_epoch+".csv")
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
        saveDir=ensure_dir_exist(
            (self.saveDir+"/"+load_model_epoch_list).replace("checkpoints","results"))
        WriteToSubmission(res,fileName=saveDir+"/predicted.csv")