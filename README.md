# TextSentimentClassification
TextSentimentClassification, using tensorflow.
# Data Preprocessing
Remove the letter whose number of repetitions is over 3 from a word...
# Word Vectors Training
Using word2vec and GloVe to generate word vectors...
![](https://github.com/wslc1314/TextSentimentClassification/blob/master/data_helpers/word2vec/model-200.png "word2vec-200")
![](https://github.com/wslc1314/TextSentimentClassification/blob/master/data_helpers/glove/model-200.png "GloVe-200")
# Models
## TextCNN
### Reference
[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
[A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1510.03820)
### Model Architecture
![](https://github.com/wslc1314/TextSentimentClassification/blob/master/images/TextCNN.png "TextCNN")
Total 4 ways:
+ CNN-rand
+ CNN-static
+ CNN-nonstatic
+ CNN-multichannel
## TextRNN
### Model Architecture
![](https://github.com/wslc1314/TextSentimentClassification/blob/master/images/TextRNN.JPG "TextRNN")
Using bidirectional RNN, and then concatenating the last output of the forward process and the first output of the backward process...
## CRNN
### Reference
[A C-LSTM Neural Network for Text Classification](https://arxiv.org/abs/1511.08630)
### Model Architecture
![](https://github.com/wslc1314/TextSentimentClassification/blob/master/images/CRNN.png "CRNN")
Using CNN to extract sentences with higher-level phrase representations, and then learning long short-term dependency with bi-RNN...
## RCNN
### Reference
[Recurrent Convolutional Neural Networks for Text Classification](https://aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9745)
### Model Architecture
![](https://github.com/wslc1314/TextSentimentClassification/blob/master/images/RCNN.png "RCNN")
In addition to implementing the same structure as the paper, using bi-LSTM or bi-GRU and then concatenating their outputs...
RNN for capturing contextual information and max pooling used for judging which words play key roles in the task...
## HAN
### Reference
[Hierarchical Attention Networks for Document Classification](https://www.microsoft.com/en-us/research/publication/hierarchical-attention-networks-document-classification/)
### Model Architecture
![](https://github.com/wslc1314/TextSentimentClassification/blob/master/images/HAN.png "HAN")
Transforming a sentence into a document consisting of sentences...
# Ensembles
## Bagging
Uniform blending...
## Stacking
Using Logistic Regression as level-2 classifier...

