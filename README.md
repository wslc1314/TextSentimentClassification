# TextSentimentClassification
TextSentimentClassification, using tensorflow.
[Original Data Achieving](https://www.kaggle.com/c/ml-2017fall-hw4/data)

# Data Preprocessing
Remove the letter whose number of repetitions is over 3 from a word...

# Word Vectors Training
Using word2vec and GloVe to generate word vectors...
![](https://github.com/wslc1314/TextSentimentClassification/blob/master/data_helpers/word2vec/model-200.png "word2vec-200")
![](https://github.com/wslc1314/TextSentimentClassification/blob/master/data_helpers/glove/model-200.png "GloVe-200")

# Models
## Performance
Model | Epoch | Training Accuracy | Validation Accuracy | Parameters(word vectors excluded)
- | :-: | :-: |  :-: | -:
TextCNN+nonstatic | 130 | 0.8839 | 0.8142 | 281,202
TextRNN+nonstatic | 150 | 0.8383 | 0.8199 | 285,826
CRNN+nonstatic | 70 | 0.8600 | 0.8219 | 274,818
RCNN+nonstatic | 50 | 0.8553 | 0.8227 | 318,978
HAN+nonstatic | 110 | 0.8355 | 0.8188 | 209,410

## TextCNN
### Reference
[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)

[A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1510.03820)
### Model Architecture
![](https://github.com/wslc1314/TextSentimentClassification/blob/master/images/TextCNN.png "TextCNN")
*Total 4 ways:*
+ CNN-rand
+ CNN-static
+ CNN-nonstatic
+ CNN-multichannel
### Performance
Model | Epoch | Training Accuracy | Validation Accuracy | Parameters(word vectors excluded)
- | :-: | :-: |  :-: | -:
TextCNN+rand | 130 | 0.8761 | 0.8137 | 281,202
TextCNN+static | 60 | 0.9015 | 0.8113 | 281,202
TextCNN+nonstatic | 130 | 0.8839 | 0.8142 | 281,202
TextCNN+multichannel | 60 | 0.9225 | 0.8141 | 561,202

Choosing to use word vectors in a nonstatic way. 

## TextRNN
### Model Architecture
![](https://github.com/wslc1314/TextSentimentClassification/blob/master/images/TextRNN.JPG "TextRNN")

Using bidirectional RNN, and then concatenating the output of the forward process and the output of the backward process...

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
Using Logistic Regression as the level-2 classifier...

## Performance
Model | Epoch | Training Accuracy | Testing Accuracy | Parameters(word vectors excluded)
- | :-: | :-: |  :-: | -:
LR+static_avg | - | 0.77364 | 0.773605 | - 
NB+static_avg | - | 0.606435 | 0.61082 | -
TextCNN+nonstatic | 130 | 0.8703 | 0.817615 | 281,202
TextRNN+nonstatic | 150 | 0.8384 | 0.81969 | 285,826
CRNN+nonstatic | 70 | 0.8589 | 0.82449 | 274,818
RCNN+nonstatic | 50 | 0.8497 | 0.822935 | 318,978
HAN+nonstatic | 110 | 0.8330 | 0.820235 | 209,410
bagging | - | 0.8538 | 0.82999 | -
stacking | - | | | -

