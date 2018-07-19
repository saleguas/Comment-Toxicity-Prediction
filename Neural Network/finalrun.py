import csv, gc, pandas as pd, numpy as np, tensorflow as tf
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GRU, Conv1D, MaxPooling1D, Bidirectional, GlobalMaxPool1D, Bidirectional
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, History
import matplotlib.pyplot as plt

classNames = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

fileNameOfTrain = "train_comment.csv"
train = pd.read_csv('../input/train.csv').fillna(' ')
fileNameOfTest = "test_comment.csv"
test = pd.read_csv('../input/test.csv').fillna(' ')

listSentencesTrain = train["comment_text"]
listSentencesTest = test["comment_text"]

tokenizer = Tokenizer(num_words = 20000, char_level = True)
tokenizer.fit_on_texts(list(listSentencesTrain))
listTokenizedTrain = tokenizer.texts_to_sequences(listSentencesTrain)
listTokenizedTest = tokenizer.texts_to_sequences(listSentencesTest)


totalNumWords = [len(commenty) for commenty in listTokenizedTrain]
print(np.percentile(totalNumWords, 75))
plt.hist(totalNumWords, bins = np.arange(0, 2000, 10))
plt.title("Frequency of Comments With a Given Tokenized Word Count in Training Data")
plt.show()
# The 75th percentile of word counts is 435, but just to be safe we will set the padding length to 450

print("Checkpoint 1")

# Padding is used because we have to feed a stream of data that has a consistent length (fixed number of features)
maxLen = 450
xTrain = pad_sequences(listTokenizedTrain, maxlen = maxLen)
xTest = pad_sequences(listTokenizedTest, maxlen = maxLen)
yTrain = train[classNames].values
input = Input(shape = (maxLen, ))

embedSize = 250
x = Embedding(len(tokenizer.word_index)+1, embedSize)(input)

print("Checkpoint 2")

x = Conv1D(filters=100, kernel_size=4, padding='same', activation='relu')(x)
x = MaxPooling1D(pool_size=4)(x)
x = Bidirectional(GRU(60, return_sequences=True, name='lstm_layer', dropout=0.2, recurrent_dropout=0.2))(x)
x = GlobalMaxPool1D()(x)

print("Checkpoint 3")

# Dropout layers disable some nodes so that the nodes in the next layer can generalize better
# The sigmoid function will squash the output between the bounds of 0 and 1

x = Dense(50, activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(6, activation="sigmoid")(x)

print("Checkpoint 4")

model = Model(inputs=input, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model.fit(xTrain, yTrain, epochs=2, verbose=2, validation_split=0.1)
print(model.summary())
print(hist.history)
yTest = model.predict(xTest, verbose=2)

print("Checkpoint 5")

fileNameOfTemplate = "sample_submission.csv"
template = pd.read_csv(fileNameOfTemplate)
submission = template
yTest[np.isnan(yTest)] = 0
submission[classNames] = yTest
submission.to_csv('submission.csv', index=False)
