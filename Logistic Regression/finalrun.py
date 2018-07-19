import numpy as np
import csv
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack

classNames = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

fileNameOfTrain = "train_comment.csv"
train = pd.read_csv(fileNameOfTrain).fillna(' ')
fileNameOfTest = "test_comment.csv"
test = pd.read_csv(fileNameOfTest).fillna(' ')

trainText = train['comment_text']
testText = test['comment_text']
allText = pd.concat([trainText, testText])

wordVectorizer = TfidfVectorizer(
    sublinear_tf = True,
    strip_accents = 'unicode',
    analyzer = 'word',
    token_pattern = r'\w{1,}',
    stop_words = 'english',
    ngram_range = (1, 1),
    max_features = 5200
)
wordVectorizer.fit(allText)
trainWordFeatures = wordVectorizer.transform(trainText)
testWordFeatures = wordVectorizer.transform(testText)

print("Checkpoint 1")

charVectorizer = TfidfVectorizer(
    sublinear_tf = True,
    strip_accents = 'unicode',
    analyzer = 'char',
    stop_words = 'english',
    ngram_range = (2, 6),
    max_features = 55000
)
charVectorizer.fit(allText)
trainCharFeatures = charVectorizer.transform(trainText)
testCharFeatures = charVectorizer.transform(testText)

print("Checkpoint 2")

trainFeatures = hstack([trainWordFeatures, trainCharFeatures])
testFeatures = hstack([testWordFeatures, testCharFeatures])

scores = []
submission = pd.DataFrame.from_dict({'id': test['id']})

print("Checkpoint 3")

for classy in classNames:
    trainTarget = train[classy]

    model = LogisticRegression(C = 0.1, solver='sag')
    cvScore = np.mean(cross_val_score(model, trainFeatures, trainTarget, scoring="roc_auc"))
    scores.append(cvScore)

    model.fit(trainFeatures, trainTarget)
    submission[classy] = model.predict_proba(testFeatures)[:, 1]
    print(classy)

for i in scores:
    print(i, end = ' ')
print()
submission.to_csv('submission.csv', index = False)
