class general_config(object):
    # path
    data_dir="data_helpers/dataset" # "Directory for saving data."
    log_dir="logs" # "Directory for saving logs and visualizing."
    save_dir="checkpoints" # "Directory for saving models."
    res_dir="results" # "Directory for saving predicted results for testing data."

    global_static_i2v_path=data_dir+"/training_testing_i2v.json"
    global_static_v2i_path=data_dir+"/training_testing_v2i.json"
    global_nonstatic_i2v_path=data_dir+"/training_i2v.json"
    global_nonstatic_v2i_path=data_dir+"/training_v2i.json"

    # model
    max_seq_len=39 # "Maximum length of a sentence."
    num_classes=2 # "The number of class."

    wv_path = "data_helpers/word2vec/model-200"  # "Path for loading word vectors."

    # train
    with_validation = True  # "Whether to do validation when training."
    load_path_train = None  # "Path for loading model when training."

    learning_rate= 0.1 # "Initial learning rate."
    min_learning_rate= 0.0005 # "Minimize learning rate."
    learning_rate_decay= 0.9 # "Learning rate decay ratio."

    num_epochs= 200 # "Total training epochs."
    steps_every_epoch= 100 # "Total training steps every epoch."
    batch_size= 128 # "Size for a batch."
    save_epochs= 10 # "Saving model every epochs."
    early_stopping= 20 # "Maximum times for accuracy in validation set is lower than current maximum value."
    num_visualize= 0 # "The number of words for visualization."

    # test
    load_path_test= None # "Path for loading model when testing."


class textcnn_config(object):
    model_type="nonstatic" # "Baseline, static, nonstatic, or multichannel."
    filter_size_list="2-3-4-5" # "String for filter size list."
    filter_num=100 # "Size for filter number."
    fc_layer_size_list=None # "String for fc layer size list."

    dropout=0. # "Probability for dropout when training."
    max_l2_norm= 10. # "Max l2-norm value for weight clipping."


class textrnn_config(object):
    cell_type="gru" # "gru or lstm."
    state_size_list = "128" # "String for state size list."
    fc_layer_size_list = "128"  # "String for fc layer size list."

    dropout = 0.  # "Probability for dropout when training."
    max_l2_norm = None  # "Max l2-norm value for weight clipping."
    grads_clip = None  # "Gradients clipping value."


class crnn_config(object):
    filter_size_list = "3"  # "String for filter size list."
    filter_num = 128  # "Size for filter number."
    cell_type = "gru"  # "gru or lstm."
    state_size_list = "128"  # "String for state size list."
    fc_layer_size_list = None  # "String for fc layer size list."

    dropout = 0.  # "Probability for dropout when training."
    max_l2_norm = None  # "Max l2-norm value for weight clipping."
    grads_clip = None  # "Gradients clipping value."
    l2_loss = 0.  # "L2-loss lambda value."


class rcnn_config(object):
    cell_type = "gru"  # "rnn, gru or lstm."
    state_size= 128  # "State size integer."
    hidden_size=256 # "Size for hidden layer."
    fc_layer_size_list = None  # "String for fc layer size list."

    dropout = 0.  # "Probability for dropout when training."
    max_l2_norm = None  # "Max l2-norm value for weight clipping."
    grads_clip = None  # "Gradients clipping value."
    l2_loss = 0.  # "L2-loss lambda value."


class han_config(object):
    state_size_word=50
    state_size_sentence=50
    attention_dim_word=100
    attention_dim_sentence=100
    fc_layer_size_list = None  # "String for fc layer size list."

    dropout = 0.  # "Probability for dropout when training."
    max_l2_norm = None  # "Max l2-norm value for weight clipping."
    grads_clip = None  # "Gradients clipping value."
    l2_loss = 0.  # "L2-loss lambda value."



"""
1: TextCNN
2: TextRNN
3: CRNN
4: RCNN
5: HAN
"""
modelDict={"1":"TextCNN",
           "2":"TextRNN",
           "3":"CRNN",
           "4":"RCNN",
           "5":"HAN"}

class bagging_config(object):
    # train
    base_model_list="-".join(['3']*10) # "String for base model list."

    # test
    load_model_epoch_list="-".join(['70']*10) # "String for load model epoch list when testing."

class stacking_config(object):
    # train
    num_cv=5 # "Size for cross validation."
    base_model_list="1-2-3-4-5"