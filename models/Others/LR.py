from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from data_helpers.utils import load_embedding_matrix,readNewFile
from configs import general_config
from utils import WriteToSubmission,my_logger,ensure_dir_exist
import numpy as np


class model(object):
    def __init__(self):
        self.embeddings=load_embedding_matrix(wv_path=general_config.wv_path,
                                              int2vocabPath=general_config.global_static_i2v_path)
        self.model=LogisticRegression()
        self.log_dir=ensure_dir_exist(general_config.log_dir+"/LR")
        self.save_dir=ensure_dir_exist(general_config.save_dir+"/LR")
        self.logger=my_logger(self.log_dir+"/log.txt")

    def train(self,
              trainPath=general_config.data_dir+"/training_label_new.txt",
              num_cv=5):
        indices, sentences, labels=readNewFile(file=trainPath,
                                               vocab2intPath=general_config.global_static_v2i_path)
        sentences_=[]
        for sentence in sentences:
            sentences_.append(self.embeddings[sentence].mean(axis=0))
        parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}  # Inverse of regularization strength
        self.model = GridSearchCV(self.model, parameters, cv=num_cv, refit=True)
        self.model.fit(X=sentences_,y=labels)
        self.logger.info(self.model.cv_results_)
        self.logger.info(self.model.get_params())
        self.logger.info("Training Accuracy: %s"%self.model.score(X=sentences_,y=labels))
        save_path = self.save_dir + "/model.pkl"
        joblib.dump(self.model, save_path)

    def test(self,
             testPath=general_config.data_dir+"/testing_data_new.txt"):
        indices, sentences, labels = readNewFile(file=testPath,
                                                 vocab2intPath=general_config.global_static_v2i_path)
        sentences_ = []
        for sentence in sentences:
            sentences_.append(self.embeddings[sentence].mean(axis=0))
        self.model = joblib.load(self.save_dir + "/model.pkl")
        predicted=self.model.predict(sentences_)
        res = np.concatenate([np.array(indices).reshape((-1, 1)),
                              np.array(predicted).reshape((-1, 1))], axis=1)
        WriteToSubmission(res, fileName=self.save_dir.replace("checkpoints", "results") + "/predicted.csv")