import tensorflow as tf
import os, numpy as np
from configs import general_config,crnn_config
from data_helpers.utils import load_embedding_matrix
from utils import PaddedDataIterator
from utils import WriteToSubmission
from utils import ensure_dir_exist,my_logger,get_num_params
from tensorflow.contrib.tensorboard.plugins import projector


class model(object):
    def __init__(self,
                 filter_size_list=crnn_config.filter_size_list,
                 filter_num=crnn_config.filter_num,
                 cell_type=crnn_config.cell_type,
                 state_size_list=crnn_config.state_size_list,
                 fc_layer_size_list=crnn_config.fc_layer_size_list,
                 dropout=crnn_config.dropout,
                 max_l2_norm=crnn_config.max_l2_norm,
                 grads_clip=crnn_config.grads_clip,
                 l2_loss=crnn_config.l2_loss,
                 wv_path=general_config.wv_path
                 ):
        self.filter_size_list = filter_size_list
        self.filter_num = filter_num
        assert cell_type in ["gru","lstm"],"Invalid cell type!"
        self.cell_type=cell_type
        self.state_size_list=state_size_list
        self.fc_layer_size_list = fc_layer_size_list
        self.dropout_value = dropout
        self.max_l2_norm = max_l2_norm
        self.grads_clip=grads_clip
        self.l2_loss=l2_loss

        self.wv_path=wv_path
        # 获得embeddings
        embeddings_ns = load_embedding_matrix(wv_path=self.wv_path,
                                              int2vocabPath=general_config.global_nonstatic_i2v_path)

        self.build_model(embeddings_ns=embeddings_ns)
        self.min_len=1+general_config.max_seq_len-max([int(i) for i in self.filter_size_list.split("-")])

    def _embedded(self, X, embeddings_ns):
        self.embedding_size = embeddings_ns.shape[1]
        self.embedding_matrix_ns = tf.get_variable(name="embedding_matrix", trainable=True,
                                                   shape=embeddings_ns.shape,
                                                   dtype=tf.float32,
                                                   initializer=tf.constant_initializer(value=embeddings_ns))
        X_embedded = tf.nn.embedding_lookup(self.embedding_matrix_ns, X)
        return X_embedded

    def _cnn(self, X_embedded):
        if self.dropout_value is not None:
            inputs = tf.nn.dropout(X_embedded, keep_prob=1 - self.dropout)
        else:
            inputs=X_embedded
        filter_size_list=[int(i) for i in self.filter_size_list.split("-")]
        h_total=[]
        for filter_size in filter_size_list:
            h = tf.layers.conv1d(
                inputs=inputs,
                filters=self.filter_num,
                kernel_size=filter_size,
                strides=1,
                padding='valid',
                data_format='channels_last',
                activation=tf.nn.relu,
                use_bias=True,
                kernel_initializer=tf.keras.initializers.he_uniform(),
                bias_initializer=tf.zeros_initializer())
            h_total.append(h)
        if len(h_total)>1:
            h=tf.concat([h[:,:self.min_len,:] for h in h_total],axis=2)
            h=tf.reshape(h,shape=[tf.shape(self.X)[0],self.min_len,self.filter_num*len(h_total)])
        else:
            h=h_total[0]
        return h

    def _rnn(self, rnn_inputs,inputs_len):
        state_size_list=[int(i) for i in self.state_size_list.split("-")]
        cells_fw = []
        cells_bw = []
        for state_size in state_size_list:
            if self.cell_type=="gru":
                cell_fw = tf.nn.rnn_cell.GRUCell(state_size,
                                                 kernel_initializer=tf.glorot_uniform_initializer(),
                                                 bias_initializer=tf.zeros_initializer())
                cell_bw = tf.nn.rnn_cell.GRUCell(state_size,
                                                 kernel_initializer=tf.glorot_uniform_initializer(),
                                                 bias_initializer=tf.zeros_initializer())
            else:
                cell_fw = tf.nn.rnn_cell.LSTMCell(state_size,
                                                 initializer=tf.glorot_uniform_initializer())
                cell_bw = tf.nn.rnn_cell.LSTMCell(state_size,
                                                 initializer=tf.glorot_uniform_initializer())
            if self.dropout_value is not None:
                cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw,output_keep_prob=1-self.dropout)
                cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, output_keep_prob=1-self.dropout)
            cells_fw.append(cell_fw)
            cells_bw.append(cell_bw)
        if len(cells_fw)>1:
            self.cell_fw = tf.nn.rnn_cell.MultiRNNCell(cells_fw)
            self.cell_bw = tf.nn.rnn_cell.MultiRNNCell(cells_bw)
        else:
            self.cell_fw=cells_fw[0]
            self.cell_bw=cells_bw[0]
        (rnn_outputs_fw, rnn_outputs_bw), final_state = \
            tf.nn.bidirectional_dynamic_rnn(cell_fw=self.cell_fw, cell_bw=self.cell_bw,
                                            inputs=rnn_inputs,sequence_length=inputs_len,
                                            dtype=tf.float32)
        rnn_outputs = tf.concat([rnn_outputs_fw, rnn_outputs_bw], axis=-1)
        rnn_outputs = tf.gather_nd(params=rnn_outputs,
                                   indices=tf.stack([tf.range(tf.shape(rnn_inputs)[0]),
                                                     inputs_len - 1], axis=1))
        return rnn_outputs

    def build_model(self, embeddings_ns=None):
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.variable_scope("input_layer"):
                self.X = tf.placeholder(dtype=tf.int32,
                                        shape=[None, general_config.max_seq_len], name="sentences_placeholder")
                self.X_len = tf.placeholder(tf.int32, shape=[None], name="sentence_lengths_placeholder")
                self.y = tf.placeholder(dtype=tf.int32,
                                        shape=[None], name="labels_placeholder")
                self.dropout = tf.placeholder_with_default(0., shape=[], name="dropout_placeholder")
                self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[], name="learning_rate_placeholder")

            with tf.variable_scope("embedded"):
                X_embedded = self._embedded(self.X, embeddings_ns)

            with tf.variable_scope("cnn_layer"):
                h=self._cnn(X_embedded)

            with tf.variable_scope("rnn_layer"):
                h=self._rnn(rnn_inputs=h,inputs_len=self.X_len)

            if self.fc_layer_size_list is not None:
                with tf.variable_scope("fc_layer"):
                    for fc_size in [int(i) for i in self.fc_layer_size_list.split("-")]:
                        h = tf.layers.dense(inputs=h, units=fc_size, activation=tf.nn.relu,
                                            kernel_initializer=tf.keras.initializers.he_uniform(),
                                            bias_initializer=tf.zeros_initializer(),
                                            kernel_regularizer=tf.keras.regularizers.l2(self.l2_loss))
                        if self.dropout_value is not None:
                            h = tf.nn.dropout(h, keep_prob=1 - self.dropout)

            with tf.variable_scope("output_layer"):
                output = tf.layers.dense(inputs=h, units=general_config.num_classes,
                                         kernel_initializer=tf.glorot_uniform_initializer(),
                                         bias_initializer=tf.zeros_initializer(),
                                         kernel_regularizer=tf.keras.regularizers.l2(self.l2_loss))
            with tf.name_scope("Loss"):
                self.loss_op = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=output),
                    reduction_indices=0)

            with tf.name_scope("Optimize"):
                optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.9)
                train_vars = tf.trainable_variables()
                if self.grads_clip is None:
                    grads = tf.gradients(self.loss_op, train_vars)
                else:
                    grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss_op, train_vars), self.grads_clip)
                self.train_op = optimizer.apply_gradients(zip(grads, train_vars))
                if self.max_l2_norm is not None:
                    clip_op = [var.assign(tf.clip_by_norm(var, clip_norm=self.max_l2_norm))
                               for var in tf.trainable_variables()]
                    self.train_op = tf.group([self.train_op, clip_op])

            with tf.name_scope("Accuracy"):
                self.predicted = tf.argmax(tf.nn.softmax(output), axis=1, output_type=tf.int32)
                correct_or_not = tf.equal(self.predicted, self.y)
                self.acc_op = tf.reduce_mean(tf.cast(correct_or_not, tf.float32))

            with tf.name_scope("Summaries"):
                loss = None
                accuracy = None
                self.loss_accuracy_summary = tf.Summary()
                self.loss_accuracy_summary.value.add(tag='Loss', simple_value=loss)
                self.loss_accuracy_summary.value.add(tag='Accuracy', simple_value=accuracy)

    def _feed_dict_train(self, batch_x, batch_y,batch_len):
        feed_dict = {self.X: batch_x, self.y: batch_y,self.X_len:batch_len,
                     self.dropout: self.dropout_value, self.learning_rate: self.learning_rate_value}
        return feed_dict

    def _feed_dict_valid(self, batch_x, batch_y,batch_len):
        feed_dict = {self.X: batch_x, self.y: batch_y,self.X_len:batch_len}
        return feed_dict

    def _feed_dict_test(self, batch_x,batch_len):
        feed_dict = {self.X: batch_x,self.X_len:batch_len}
        return feed_dict


    def fit(self,trainFile=None,with_validation=general_config.with_validation,
              log_dir=general_config.log_dir+"/CRNN",
              save_dir=general_config.save_dir+"/CRNN",
              load_path=general_config.load_path_train,
              num_epochs=general_config.num_epochs, steps_every_epoch=general_config.steps_every_epoch,
            batch_size=general_config.batch_size,
              learning_rate=general_config.learning_rate,
              lr_changing=general_config.lr_changing,
              min_learning_rate=general_config.min_learning_rate,
            learning_rate_decay=general_config.learning_rate_decay,
            save_epochs=general_config.save_epochs,early_stopping=general_config.early_stopping,
            num_visual=general_config.num_visualize):
        self.learning_rate_value=learning_rate
        self.trainFile = trainFile
        self.validFile = None
        self.with_validation = with_validation
        if self.trainFile is None:
            if self.with_validation:
                self.trainFile = general_config.train_file
            else:
                self.trainFile = general_config.training_file
        if self.with_validation:
            self.validFile = self.trainFile.replace("train", "valid")
        tmp = os.path.join(os.path.dirname(self.trainFile),
                           os.path.basename(self.trainFile).replace(".txt", "").split("_")[0])
        self.int2vocabPath = tmp + "_i2v.json"
        self.vocab2intPath = tmp + "_v2i.json"
        metadataPath = {}
        metadataPath["nonstatic"] = "/home/leechen/code/python/TextSentimentClassification/" \
                                    + self.vocab2intPath.replace("v2i.json", "metadata.tsv")

        train_loss = []
        train_accuracy = []
        valid_loss = []
        valid_accuracy = []
        # 训练过程中的日志保存文件以及模型保存路径
        if self.with_validation:
            log_dir=ensure_dir_exist(log_dir+"/train_valid")
            train_dir = os.path.join(log_dir, "train")
            val_dir = os.path.join(log_dir, "valid")
            save_dir = ensure_dir_exist(save_dir + "/train_valid")
        else:
            log_dir=ensure_dir_exist(log_dir+"/train")
            train_dir = os.path.join(log_dir, "train")
            val_dir=None
            save_dir = ensure_dir_exist(save_dir + "/train")

        # 生成日志
        logger=my_logger(log_dir+"/log_fit.txt")
        msg = "\n--filter_size_list: %s\n" % self.filter_size_list \
              + "--filter_num: %s\n" % self.filter_num\
              + "--cell_type: %s\n" % self.cell_type \
              + "--state_size_list: %s\n" % self.state_size_list \
              + "--fc_layer_size_list: %s\n" % self.fc_layer_size_list\
              + "--embedding_size: %s\n" % self.embedding_size \
              + "--dropout: %s\n" % self.dropout_value \
              + "--max_l2_norm: %s\n" % self.max_l2_norm \
              + "--grads_clip: %s\n" % self.grads_clip \
              + "--l2_loss: %s\n" % self.l2_loss \
              + "--learning_rate: %s\n" % self.learning_rate_value\
              +"--lr_changing: %s\n"%lr_changing\
              + "--min_learning_rate: %s\n" % min_learning_rate\
              + "--learning_rate_decay: %s\n" % learning_rate_decay\
              +"--load_path: %s\n" % load_path\
              +"--num_epochs: %s\n" % num_epochs\
              +"--steps_every_epoch: %s\n" % steps_every_epoch\
              +"--batch_size: %s\n" % batch_size\
              +"--save_epochs: %s\n" % save_epochs\
              +"--early_stopping: %s\n" % early_stopping\
              +"--num_visual: %s"%num_visual
        logger.info(msg)

        # 定义数据生成器
        train_generator = PaddedDataIterator(loadPath=self.trainFile,vocab2intPath=self.vocab2intPath,
                                             sent_len_cut=self.min_len)
        val_generator = None if self.validFile is None else PaddedDataIterator(loadPath=self.validFile,
                                                                               vocab2intPath=self.vocab2intPath,
                                                                               sent_len_cut=self.min_len)

        os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.8

        with tf.Session(config=config,graph=self.graph) as sess:
            train_writer = tf.summary.FileWriter(train_dir, sess.graph)
            val_writer = None if val_dir is None else tf.summary.FileWriter(val_dir)
            saver = tf.train.Saver(max_to_keep=5)
            sess.run(tf.global_variables_initializer())
            start = 0
            if isinstance(load_path, str):
                if os.path.isdir(load_path):
                    ckpt = tf.train.get_checkpoint_state(load_path)
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    start = ckpt.model_checkpoint_path.split("-")[-1]
                else:
                    saver.restore(sess, load_path)
                    start = load_path.split("-")[-1]
                logger.info("Loading successfully, loading epoch is %s" % start)
            logger.info("The total number of trainable variables: %s" % get_num_params())
            cur_early_stopping = 0
            cur_max_acc = 0.

            logger.info('******* start training with %d *******' % start)
            epoch=0
            for epoch in range(start, num_epochs):
                if lr_changing:
                    try:
                        if (train_loss[-1]>train_loss[-2]):
                            tmp=self.learning_rate_value*learning_rate_decay
                            if (tmp>=min_learning_rate):
                                self.learning_rate_value=tmp
                                logger.info("Learning rate multiplied by %s at epoch %s."%(learning_rate_decay,epoch+1))
                        else:
                            if (train_loss[-1]<train_loss[-2]-0.015):
                                self.learning_rate_value*=1.05
                                logger.info("Learning rate multiplied by 1.05 at epoch %s."%(epoch+1))
                    except:
                        pass

                avg_loss_t, avg_accuracy_t = 0, 0
                avg_loss_v, avg_accuracy_v = 0, 0
                for step in range(steps_every_epoch):
                    _, batch_seqs, batch_labels, batch_lens = train_generator.next(batch_size)
                    sess.run(self.train_op,
                             feed_dict=self._feed_dict_train(batch_x=batch_seqs, batch_y=batch_labels,
                                                             batch_len=batch_lens))
                    loss_t, acc_t= sess.run([self.loss_op, self.acc_op],
                        feed_dict=self._feed_dict_valid(batch_x=batch_seqs, batch_y=batch_labels,
                                                        batch_len=batch_lens))
                    avg_loss_t += loss_t
                    avg_accuracy_t += acc_t
                avg_loss_t/=steps_every_epoch
                avg_accuracy_t/=steps_every_epoch
                train_loss.append(avg_loss_t)
                train_accuracy.append(avg_accuracy_t)
                self.loss_accuracy_summary.value[0].simple_value = avg_loss_t
                self.loss_accuracy_summary.value[1].simple_value = avg_accuracy_t
                train_writer.add_summary(summary=self.loss_accuracy_summary, global_step=epoch + 1)
                if self.with_validation:
                   # 计算验证集上的表现
                    cur_loop=val_generator.loop
                    _, batch_seqs, batch_labels,batch_lens = val_generator.next(1024,need_all=True)
                    cur_count=0
                    while(val_generator.loop==cur_loop):
                        loss_v, acc_v = sess.run([self.loss_op, self.acc_op],
                                                 feed_dict=self._feed_dict_valid(batch_seqs, batch_labels,
                                                                                 batch_lens))
                        avg_loss_v += loss_v
                        avg_accuracy_v += acc_v
                        cur_count += 1
                        _, batch_seqs, batch_labels, batch_lens= val_generator.next(1024, need_all=True)
                    avg_loss_v/=cur_count
                    avg_accuracy_v/=cur_count
                    valid_loss.append(avg_loss_v)
                    valid_accuracy.append(avg_accuracy_v)
                    self.loss_accuracy_summary.value[0].simple_value = avg_loss_v
                    self.loss_accuracy_summary.value[1].simple_value = avg_accuracy_v
                    val_writer.add_summary(summary=self.loss_accuracy_summary, global_step=epoch + 1)
                    logger.info("Epoch: [%04d/%04d], "
                          "Training Loss: %.4f, Training Accuracy: %.4f, "
                          "Validation Loss: %.4f, Validation Accuracy: %.4f" \
                          % (epoch + 1, num_epochs,
                             avg_loss_t, avg_accuracy_t, avg_loss_v, avg_accuracy_v))

                   # 如果验证集上的准确率连续低于历史最高准确率的次数超过early_stopping次，则提前停止迭代。
                    if (avg_accuracy_v > cur_max_acc):
                        cur_max_acc = avg_accuracy_v
                        cur_early_stopping = 0
                        logger.info("Saving model-%s" % (epoch + 1))
                        saver.save(sess, os.path.join(save_dir, 'model.ckpt'), global_step=epoch + 1)
                    else:
                        cur_early_stopping += 1
                    if cur_early_stopping > early_stopping:
                        logger.info("Early stopping after epoch %s !" % (epoch + 1))
                        break
                else:
                    logger.info("Epoch: [%04d/%04d], "
                                "Training Loss: %.4f, Training Accuracy: %.4f " \
                                % (epoch + 1, num_epochs,avg_loss_t, avg_accuracy_t))
                # 保存一次模型
                if (epoch - start + 1) % save_epochs == 0:
                    logger.info("Saving model-%s"%(epoch+1))
                    saver.save(sess, os.path.join(save_dir, 'model.ckpt'), global_step=epoch + 1)
            if num_visual > 0:
                # 可视化最终词向量
                config = projector.ProjectorConfig()
                final_embeddings = {}
                try:
                    final_embeddings["nonstatic"] = self.embedding_matrix_ns.eval()[:num_visual]
                except:
                    pass
                for (name, final_embedding) in final_embeddings.items():
                    embedding_var = tf.Variable(final_embedding, name="word_embeddings_" + name)
                    sess.run(embedding_var.initializer)
                    saver = tf.train.Saver([embedding_var])
                    saver.save(sess, log_dir + "/embeddings_" + name + ".ckpt-" + str(epoch+1))
                    embedding = config.embeddings.add()
                    embedding.tensor_name = embedding_var.name
                    embedding.metadata_path = metadataPath[name]
                projector.visualize_embeddings(train_writer, config)
        return train_loss, train_accuracy, valid_loss, valid_accuracy

    def evaluate(self, load_path=general_config.load_path_test,
                 validFile=None, vocab2intPath=None):
        if validFile is None or vocab2intPath is None:
            validFile = general_config.training_file
            vocab2intPath = general_config.global_nonstatic_v2i_path
        
        train_generator = PaddedDataIterator(loadPath=validFile,
                                             vocab2intPath=vocab2intPath,
                                             sent_len_cut=self.min_len)
        load_dir = load_path if os.path.isdir(load_path) else os.path.dirname(load_path)
        log_dir = load_dir.replace("checkpoints", "logs")
        logger = my_logger(log_dir + "/log_evaluate.txt")

        os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.8

        with tf.Session(config=config, graph=self.graph) as sess:
            logger.info("Loading model...")
            saver = tf.train.Saver()
            if os.path.isdir(load_path):
                ckpt = tf.train.get_checkpoint_state(load_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split("-")[-1]
            else:
                saver.restore(sess, load_path)
                global_step = load_path.split("-")[-1]
            logger.info("Loading successfully, loading epoch is %s" % global_step)
            logger.info("The total number of trainable variables: %s" % get_num_params())

            cur_loop = train_generator.loop
            cur_count = 0
            avg_loss_t, avg_accuracy_t = 0., 0.
            _, batch_seqs, batch_labels, batch_lens = train_generator.next(1024, need_all=True)
            while (train_generator.loop == cur_loop):
                cur_count += 1
                loss_t, acc_t = sess.run([self.loss_op, self.acc_op],
                                         feed_dict=self._feed_dict_valid(batch_seqs, batch_labels,batch_lens))
                avg_loss_t += loss_t
                avg_accuracy_t += acc_t
                _, batch_seqs, batch_labels, batch_lens= train_generator.next(1024, need_all=True)
            avg_loss_t/=cur_count
            avg_accuracy_t/=cur_count
            logger.info("Loss: %.4f, Accuracy: %.4f "% (avg_loss_t, avg_accuracy_t))
        return avg_loss_t, avg_accuracy_t

    def predict(self, testFile=None,vocab2intPath=None,
             load_path=general_config.load_path_test,
             is_save=True,resPath=None):

        if testFile is None or vocab2intPath is None:
            testFile=os.path.join(general_config.data_dir,"testing_data_new.txt")
            vocab2intPath=general_config.global_nonstatic_v2i_path
        test_generator = PaddedDataIterator(loadPath=testFile,vocab2intPath=vocab2intPath,
                                            sent_len_cut=self.min_len)
        load_dir = load_path if os.path.isdir(load_path) else os.path.dirname(load_path)
        log_dir = load_dir.replace("checkpoints", "logs")
        logger = my_logger(log_dir + "/log_predict.txt")
        os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        with tf.Session(config=config,graph=self.graph) as sess:
            logger.info("Loading model...")
            saver = tf.train.Saver()
            if os.path.isdir(load_path):
                ckpt = tf.train.get_checkpoint_state(load_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split("-")[-1]
            else:
                saver.restore(sess, load_path)
                global_step = load_path.split("-")[-1]
            logger.info("Loading successfully, loading epoch is %s" % global_step)

            cur_loop = test_generator.loop
            batch_idx, batch_seqs, _, batch_lens= test_generator.next(batch_size=1024, need_all=True)
            res={}
            while (test_generator.loop == cur_loop):
                predicted = sess.run(self.predicted,
                                     feed_dict=self._feed_dict_test(batch_seqs,batch_lens))
                for (id, label) in zip(batch_idx, predicted):
                    res[id] = int(label)
                batch_idx, batch_seqs, _, batch_lens = test_generator.next(1024, need_all=True)
            if is_save:
                if resPath is None:
                    res_dir = ensure_dir_exist(load_dir.replace("checkpoints", "results"))
                    resPath = os.path.join(res_dir, "predicted.csv-" + str(global_step))
                res_save = [[key, value] for (key, value) in res.items()]
                # 用于存放测试识别结果
                WriteToSubmission(fileName=resPath, res=res_save)
        return res