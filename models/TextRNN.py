import tensorflow as tf
from configs import general_config,textrnn_config
import os
from data_helpers.utils import load_embedding_matrix
from utils import BucketedDataIterator
from utils import WriteToSubmission
from utils import ensure_dir_exist
import logging
from tensorflow.contrib.tensorboard.plugins import projector


class model(object):

    def __init__(self,
                 cell_type=textrnn_config.cell_type,
                 state_size_list=textrnn_config.state_size_list,
                 fc_layer_size_list=textrnn_config.fc_layer_size_list,
                 dropout=textrnn_config.dropout,
                 max_l2_norm=textrnn_config.max_l2_norm,
                 grads_clip = textrnn_config.grads_clip,
                 wv_path=general_config.wv_path
                 ):
        assert cell_type in ["gru","lstm"],"Invalid cell type!"
        self.cell_type=cell_type
        self.state_size_list = state_size_list
        self.fc_layer_size_list = fc_layer_size_list
        self.dropout_value = dropout
        self.max_l2_norm = max_l2_norm
        self.grads_clip=grads_clip
        
        self.wv_path=wv_path
        # 获得embeddings
        embeddings_ns = load_embedding_matrix(wv_path=self.wv_path,
                                              int2vocabPath=general_config.global_nonstatic_i2v_path)

        self.build_model(embeddings_ns=embeddings_ns)

    def _embedded(self,embeddings_ns):
        self.embedding_size = embeddings_ns.shape[1]
        self.embedding_matrix_ns = tf.get_variable(name="embedding_matrix", trainable=True,
                                                   shape=embeddings_ns.shape,
                                                   dtype=tf.float32,
                                                   initializer=tf.constant_initializer(value=embeddings_ns))
        X_embedded = tf.nn.embedding_lookup(self.embedding_matrix_ns, self.X)
        return  X_embedded

    def _rnn(self, rnn_inputs):
        state_size_list = [int(i) for i in self.state_size_list.split("-")]
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
                cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, output_keep_prob=1 - self.dropout)
                cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, output_keep_prob=1 - self.dropout)
            cells_fw.append(cell_fw)
            cells_bw.append(cell_bw)
        if len(cells_fw) > 1:
            self.cell_fw = tf.nn.rnn_cell.MultiRNNCell(cells_fw)
            self.cell_bw = tf.nn.rnn_cell.MultiRNNCell(cells_bw)
        else:
            self.cell_fw = cells_fw[0]
            self.cell_bw = cells_bw[0]
        (rnn_outputs_fw, rnn_outputs_bw), final_state = \
            tf.nn.bidirectional_dynamic_rnn(cell_fw=self.cell_fw, cell_bw=self.cell_bw,
                                            inputs=rnn_inputs, sequence_length=self.X_len,
                                            dtype=tf.float32)

        rnn_outputs_fw = tf.gather_nd(params=rnn_outputs_fw,
                                      indices=tf.stack([tf.range(tf.shape(self.X)[0]),
                                                        self.X_len - 1], axis=1))
        rnn_outputs_bw = tf.gather_nd(params=rnn_outputs_bw,
                                      indices=tf.stack([tf.range(tf.shape(self.X)[0]),
                                                        tf.zeros([tf.shape(self.X)[0]],
                                                                 dtype=tf.int32)], axis=1))
        rnn_outputs = tf.concat([rnn_outputs_fw, rnn_outputs_bw], axis=-1)
        return rnn_outputs

    def build_model(self,embeddings_ns):
        self.graph=tf.Graph()
        with self.graph.as_default():
            with tf.variable_scope("input_layer"):
                self.X = tf.placeholder(tf.int32, shape=[None, None], name="sentences_placeholder")
                self.y = tf.placeholder(tf.int32, shape=[None], name="labels_placeholder")
                self.X_len = tf.placeholder(tf.int32, shape=[None], name="sentence_lengths_placeholder")

                self.dropout = tf.placeholder_with_default(0., shape=[], name="dropout_placeholder")
                self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[], name="learning_rate_placeholder")

            with tf.variable_scope("embedded"):
                inputs = self._embedded(embeddings_ns)

            with tf.variable_scope("rnn_layer"):
                h = self._rnn(rnn_inputs=inputs)

            if self.fc_layer_size_list is not None:
                with tf.variable_scope("fc_layer"):
                    for fc_size in [int(i) for i in self.fc_layer_size_list.split("-")]:
                        h = tf.layers.dense(inputs=h, units=fc_size, activation=tf.nn.relu,
                                            kernel_initializer=tf.keras.initializers.he_uniform(),
                                            bias_initializer=tf.zeros_initializer())
                        if self.dropout_value is not None:
                            h = tf.nn.dropout(h, keep_prob=1-self.dropout)

            with tf.variable_scope("output_layer"):
                output = tf.layers.dense(inputs=h, units=general_config.num_classes,
                                         kernel_initializer=tf.glorot_uniform_initializer(),
                                         bias_initializer=tf.zeros_initializer())

            with tf.name_scope("Loss"):
                self.loss_op = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=output),
                    reduction_indices=0)

            with tf.name_scope("Optimize"):
                optimizer = tf.train.MomentumOptimizer(self.learning_rate,momentum=0.9)
                train_vars = tf.trainable_variables()
                if self.grads_clip is None:
                    grads= tf.gradients(self.loss_op, train_vars)
                else:
                    grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss_op, train_vars), self.grads_clip)
                self.train_op = optimizer.apply_gradients(zip(grads, train_vars))
                if self.max_l2_norm is not None:
                    clip_op=[var.assign(tf.clip_by_norm(var, clip_norm=self.max_l2_norm))
                                for var in tf.trainable_variables()]
                    self.train_op=tf.group([self.train_op,clip_op])

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

    def _feed_dict_train(self, batch_x, batch_y, batch_len):
        feed_dict = {self.X: batch_x, self.y: batch_y, self.X_len: batch_len,
                     self.dropout: self.dropout_value, self.learning_rate: self.learning_rate_value}
        return feed_dict

    def _feed_dict_valid(self, batch_x, batch_y, batch_len):
        feed_dict = {self.X: batch_x, self.y: batch_y, self.X_len: batch_len}
        return feed_dict

    def _feed_dict_test(self, batch_x, batch_len):
        feed_dict = {self.X: batch_x, self.X_len: batch_len}
        return feed_dict

    def train(self,trainFile=None,with_validation=general_config.with_validation,
              log_dir=general_config.log_dir+"/TextRNN",
              save_dir=general_config.save_dir+"/TextRNN",load_path=general_config.load_path_train,
            num_epochs=general_config.num_epochs, steps_every_epoch=general_config.steps_every_epoch,
            batch_size=general_config.batch_size,
              learning_rate=general_config.learning_rate,
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
                self.trainFile = general_config.data_dir + "/train.txt"
            else:
                self.trainFile = general_config.data_dir + "/training_label_new.txt"
        if self.with_validation:
            self.validFile = self.trainFile.replace("train", "valid")
        tmp = os.path.join(os.path.dirname(self.trainFile),
                           os.path.basename(self.trainFile).replace(".txt", "").split("_")[0])
        self.int2vocabPath = tmp + "_i2v.json"
        self.vocab2intPath = tmp + "_v2i.json"
        self.metadataPath = {}
        self.metadataPath["nonstatic"] = "/home/leechen/code/python/TextSentimentClassification/" \
                                         + self.int2vocabPath.replace("i2v.json", "metadata.tsv")
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
        logging_path = os.path.join(log_dir, "log.txt")
        logger = logging.getLogger(__name__)
        logger.setLevel(level=logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        logger.handlers = []
        assert len(logger.handlers) == 0
        handler = logging.FileHandler(logging_path)
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        # console.setFormatter(formatter)
        logger.addHandler(handler)
        logger.addHandler(console)
        msg = "\n--cell_type: %s\n" % self.cell_type \
              +"--state_size_list: %s\n" % self.state_size_list \
              + "--fc_layer_size_list: %s\n" % self.fc_layer_size_list \
              + "--embedding_size: %s\n" % self.embedding_size \
              + "--dropout: %s\n" % self.dropout_value \
              + "--max_l2_norm: %s\n" % self.max_l2_norm\
              +"--grads_clip: %s\n"%self.grads_clip\
              + "--learning_rate: %s\n" % self.learning_rate_value \
              + "--min_learning_rate: %s\n" % min_learning_rate\
              + "--learning_rate_decay: %s\n" % learning_rate_decay\
              +"--load_path: %s\n" % load_path\
              +"--num_epochs: %s\n" % num_epochs\
              +"--steps_every_epoch: %s\n" % steps_every_epoch\
              +"--batch_size: %s\n" % batch_size\
              +"--save_epochs: %s\n" % save_epochs\
              +"--early_stopping: %s\n" % early_stopping\
              +"--num_visual: %s\n" % num_visual
        logger.info(msg)

        # 定义数据生成器
        train_generator = BucketedDataIterator(loadPath=self.trainFile,vocab2intPath=self.vocab2intPath)
        val_generator = None if self.validFile is None else BucketedDataIterator(loadPath=self.validFile,
                                                                                 vocab2intPath=self.vocab2intPath)

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
                logger.info("Reading checkpoints...")
                saver.restore(sess, load_path)
                start=int(load_path.split("-")[-1])
                logger.info("Loading successfully, loading epoch is %s" % start)

            cur_early_stopping=0
            cur_max_acc=0.
            logger.info('******* start with %d *******' % start)
            for epoch in range(start, num_epochs):
                try:
                    if (train_loss[-1]>train_loss[-2]):
                        tmp=self.learning_rate_value*learning_rate_decay
                        if (tmp>=min_learning_rate):
                            self.learning_rate_value=tmp
                            logger.info("Learning rate multiplied by %s at epoch %s."
                                        %(learning_rate_decay,epoch+1))
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
                             feed_dict=self._feed_dict_train(batch_x=batch_seqs, batch_y=batch_labels,batch_len=batch_lens))
                    loss_t, acc_t= sess.run([self.loss_op, self.acc_op],
                        feed_dict=self._feed_dict_valid(batch_x=batch_seqs, batch_y=batch_labels,batch_len=batch_lens))
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
                                                 feed_dict=self._feed_dict_valid(batch_seqs, batch_labels,batch_lens))
                        avg_loss_v += loss_v
                        avg_accuracy_v += acc_v
                        cur_count += 1
                        _, batch_seqs, batch_labels, batch_lens = val_generator.next(1024, need_all=True)
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
                final_embeddings["nonstatic"] = self.embedding_matrix_ns.eval()[:num_visual]
                for (name, final_embedding) in final_embeddings.items():
                    embedding_var = tf.Variable(final_embedding, name="word_embeddings_" + name)
                    sess.run(embedding_var.initializer)
                    saver = tf.train.Saver([embedding_var])
                    saver.save(sess, log_dir + "/embeddings.ckpt-" + name)
                    embedding = config.embeddings.add()
                    embedding.tensor_name = embedding_var.name
                    embedding.metadata_path = self.metadataPath[name]
                projector.visualize_embeddings(train_writer, config)
        return train_loss, train_accuracy, valid_loss, valid_accuracy

    def test(self, testFile=None,vocab2intPath=None,
             load_path=general_config.load_path_test,resPath=None):
        if testFile is None or vocab2intPath is None:
            testFile=os.path.join(general_config.data_dir,"testing_data_new.txt")
            vocab2intPath=general_config.global_nonstatic_v2i_path
        test_generator = BucketedDataIterator(loadPath=testFile,vocab2intPath=vocab2intPath)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        with tf.Session(config=config,graph=self.graph) as sess:
            print("Loading model...")
            saver = tf.train.Saver()
            if os.path.isfile(load_path):
                saver.restore(sess, load_path)
                global_step = load_path.split("-")[-1]
            else:
                ckpt = tf.train.get_checkpoint_state(load_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split("-")[-1]
            print("Loading successfully, loading epoch is %s" % global_step)

            cur_loop = test_generator.loop
            batch_idx, batch_seqs, _, batch_lens = test_generator.next(batch_size=1024, need_all=True)
            res={}
            while (test_generator.loop == cur_loop):
                predicted = sess.run(self.predicted,
                                     feed_dict=self._feed_dict_test(batch_seqs,batch_lens))
                for (id, label) in zip(batch_idx, predicted):
                    res[id] = int(label)
                batch_idx, batch_seqs, _, batch_lens = test_generator.next(1024, need_all=True)
            if resPath is None:
                res_dir = ensure_dir_exist(os.path.dirname(load_path).replace("checkpoints", "results"))
                resPath = os.path.join(res_dir, "predicted_" + str(global_step) + ".csv")
            res_save = [[key, value] for (key, value) in res.items()]
            # 用于存放测试识别结果
            WriteToSubmission(fileName=resPath, res=res_save)
        return res