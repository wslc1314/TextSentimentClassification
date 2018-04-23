from configs import general_config,stacking_config,modelDict
from sklearn.model_selection import KFold,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from models.TextCNN import model as TextCNN
from models.TextRNN import model as TextRNN
from models.CRNN import model as CRNN
from models.RCNN import model as RCNN
from models.HAN import model as HAN
from utils import ensure_dir_exist,WriteToSubmission,my_logger
from data_helpers.utils import getNonstaticWordDict,create_visual_metadata,readNewFile
import numpy as np, pandas as pd
import os


def createCrossValidationData(num_cv=5):
    trainingFile=general_config.training_file
    with open(trainingFile,'r') as f:
        raw_data=np.asarray(f.readlines())
    saveDir=ensure_dir_exist(general_config.data_dir+"/cv/"+str(num_cv))
    kf=KFold(num_cv,random_state=1234+num_cv,shuffle=True)
    count=0
    for train_index,test_index in kf.split(raw_data):
        train=raw_data[train_index]
        test=raw_data[test_index]
        with open(saveDir+"/train"+str(count)+".txt",'w') as f:
            f.writelines(train)
        getNonstaticWordDict(trainFile=saveDir+"/train"+str(count)+".txt",
                             global_v2i_path=general_config.global_nonstatic_v2i_path)
        create_visual_metadata(int2vocab_path=saveDir+"/train"+str(count)+"_i2v.json")
        with open(saveDir+"/valid"+str(count)+".txt",'w') as f:
            f.writelines(test)
        count+=1


class model(object):

    def __init__(self,
                 base_model_list=stacking_config.base_model_list,
                 num_cv=stacking_config.num_cv):
        self.base_model_list = base_model_list.split("-")
        self.num_models=len(self.base_model_list)
        self.num_cv=num_cv
        self.dataDir = general_config.data_dir + "/cv/" + str(self.num_cv)
        if not os.path.exists(self.dataDir):
            createCrossValidationData(self.num_cv)

        self.models = []
        self.models_name = []
        for n in range(self.num_models):
            base_model = self.base_model_list[n]
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
        self.logDir = ensure_dir_exist(general_config.log_dir + "/stacking/"
                                       + "-".join(self.models_name)+"/"+str(self.num_cv))
        self.saveDir = ensure_dir_exist(general_config.save_dir + "/stacking/"
                                        + "-".join(self.models_name)+"/"+str(self.num_cv))
        self.classifier=LogisticRegression()
        self.logger=my_logger(self.logDir+"/log.txt")

    # level-l train
    def train_1(self):
        for i in range(self.num_models):
            model=self.models[i]
            model_name=self.models_name[i]
            log_dir_tmp=self.logDir + "/" + model_name
            save_dir_tmp=self.saveDir+"/"+model_name
            for i in range(self.num_cv):
                log_dir=log_dir_tmp+"/"+str(i)
                save_dir=save_dir_tmp+"/"+str(i)
                trainFile = self.dataDir + "/train" + str(i) + ".txt"
                if not os.path.exists(save_dir):
                    model.fit(trainFile=trainFile,with_validation=True,
                              log_dir=log_dir,save_dir=save_dir,
                              num_visual=0)

    # level-2 train
    def train_2(self):
        predicted_train=None
        id_train=None
        for i in range(self.num_models):
            model=self.models[i]
            model_name=self.models_name[i]
            save_dir_tmp=self.saveDir+"/"+model_name
            res={}
            for i in range(self.num_cv):
                save_dir=save_dir_tmp+"/"+str(i)
                if model_name=="TextCNN":
                    save_dir+="/nonstatic"
                save_dir+="/train_valid"
                testFile = self.dataDir + "/valid" + str(i) + ".txt"
                vocab2intPath=testFile.replace("valid","train").replace(".txt","_v2i.json")
                resPath=save_dir + "/valid_predicted.csv"
                if os.path.exists(resPath):
                    res_={}
                    res_tmp=pd.read_csv(filepath_or_buffer=resPath)
                    for id,label in zip(res_tmp["id"].values,res_tmp["label"].values):
                        res_[id]=label
                else:
                    res_=model.test(testFile=testFile,vocab2intPath=vocab2intPath,
                                load_path=save_dir,
                                resPath=resPath)
                res.update(res_)
            res = [[key, value] for (key, value) in res.items()]
            tmp = pd.DataFrame(res, columns=["id", "label"])
            tmp = tmp.sort_values(by="id", axis=0, ascending=True)
            id_train=np.reshape(tmp["id"].values,newshape=(-1,))
            try:
                predicted_train=np.concatenate([predicted_train,tmp["label"].values.reshape((-1,1))],
                                               axis=-1)
            except:
                predicted_train=tmp["label"].values.reshape((-1,1))
        assert predicted_train.shape[1]==self.num_models
        id,_,label=readNewFile(file=general_config.training_file)
        assert np.allclose(np.array(id),np.array(id_train)),"Inconsistent indices!"
        parameters = {'C': [0.001,0.01,0.1,1,10,100]}# Inverse of regularization strength;
        # must be a positive float.
        # Like in support vector machines, smaller values specify stronger regularization.
        self.classifier = GridSearchCV(self.classifier, parameters,cv=self.num_cv,refit=True)
        self.classifier.fit(predicted_train,np.array(label))
        self.logger.info(self.classifier.cv_results_)
        self.logger.info(self.classifier.get_params())
        save_path=self.saveDir+"/lr.pkl"
        joblib.dump(self.classifier, save_path)

    def evaluate(self,validFile=None):
        if validFile is None:
            trainFile=general_config.training_file
        else:
            trainFile=validFile
        predicted_train = None
        id_train = None
        for i in range(self.num_models):
            model = self.models[i]
            model_name = self.models_name[i]
            save_dir_tmp = self.saveDir + "/" + model_name
            res_ = None
            for i in range(self.num_cv):
                save_dir = save_dir_tmp + "/" + str(i)
                if model_name == "TextCNN":
                    save_dir += "/nonstatic"
                save_dir += "/train_valid"
                vocab2intPath = (self.dataDir + "/train" + str(i) + ".txt").replace(".txt", "_v2i.json")
                resPath = save_dir + "/train_predicted.csv"
                if os.path.exists(resPath):
                    res_ = {}
                    res_tmp = pd.read_csv(filepath_or_buffer=resPath)
                    for id, label in zip(res_tmp["id"].values, res_tmp["label"].values):
                        res_[id] = label
                else:
                    res_ = model.predict(testFile=trainFile, vocab2intPath=vocab2intPath,
                                      load_path=save_dir, resPath=resPath)
                    res_ = [[key, value] for (key, value) in res_.items()]
                    tmp = pd.DataFrame(res_, columns=["id", "label"])
                    tmp = tmp.sort_values(by="id", axis=0, ascending=True)
                    if i == 0:
                        id_train = tmp["id"].values
                    else:
                        assert np.allclose(id_train, tmp["id"].values)
                    try:
                        res_ += tmp["label"].values
                    except:
                        res_ = tmp["label"].values
            res_ = res_ / self.num_cv
            try:
                predicted_train = np.concatenate([predicted_train, res_.reshape((-1, 1))], axis=-1)
            except:
                predicted_train = res_.reshape((-1, 1))
        assert predicted_train.shape[1] == self.num_models
        predicted_ = self.classifier.predict(predicted_train)
        _, _, label = readNewFile(trainFile)
        train_accuracy = np.mean(np.equal(np.array(label).reshape((-1,))
                                          , np.array(predicted_).reshape((-1,))), axis=0)
        self.logger.info("Accuracy: %s" % train_accuracy)
        return train_accuracy

    def predict(self,testFile=None):
        if testFile is None:
            testFile=general_config.testing_file
        predicted_test = None
        id_test=None
        for i in range(self.num_models):
            model=self.models[i]
            model_name=self.models_name[i]
            save_dir_tmp = self.saveDir + "/" + model_name
            res = None
            for i in range(self.num_cv):
                save_dir =save_dir_tmp+ "/" + str(i)
                if model_name=="TextCNN":
                    save_dir+="/nonstatic"
                save_dir+="/train_valid"
                vocab2intPath =(self.dataDir + "/train" + str(i) + ".txt").replace(".txt", "_v2i.json")
                resPath=save_dir+"/test_predicted.csv"
                if os.path.exists(resPath):
                    res_={}
                    res_tmp=pd.read_csv(filepath_or_buffer=resPath)
                    for id,label in zip(res_tmp["id"].values,res_tmp["label"].values):
                        res_[id]=label
                else:
                    res_ = model.test(testFile=testFile, vocab2intPath=vocab2intPath,
                                      load_path=save_dir,
                                      resPath=resPath)
                res_ = [[key, value] for (key, value) in res_.items()]
                tmp = pd.DataFrame(res_, columns=["id", "label"])
                tmp = tmp.sort_values(by="id", axis=0, ascending=True)
                if i==0:
                    id_test=tmp["id"].values
                else:
                    assert np.allclose(id_test,tmp["id"].values)
                try:
                    res += tmp["label"].values
                except:
                    res = tmp["label"].values
            res =res/self.num_cv
            try:
                predicted_test=np.concatenate([predicted_test,res.reshape((-1,1))],axis=-1)
            except:
                predicted_test=res.reshape((-1,1))
        assert predicted_test.shape[1]==self.num_models
        self.classifier = joblib.load(self.saveDir+"/lr.pkl")
        predicted=self.classifier.predict(predicted_test)
        res=np.concatenate([id_test.reshape((-1,1)),predicted.reshape((-1,1))],axis=1)
        WriteToSubmission(res,fileName=self.saveDir.replace("checkpoints","results")+"/predicted.csv")