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

print("Checkpoint 1")

charVectorizer = TfidfVectorizer(
    sublinear_tf = True,
    strip_accents = 'unicode',
    analyzer = 'char',
    stop_words = 'english',
    ngram_range = (2, 6),
    max_features = 50000
)
charVectorizer.fit(allText)
trainCharFeatures = charVectorizer.transform(trainText)

print("Checkpoint 2")

lo = 1000
hi = 20000
bestScore = 0
bestW = 0
iterCount = 0

while lo < hi:
    t1 = (lo*2+hi)//3
    t2 = (lo+hi*2)//3
    iterCount += 1
    print("Iteration " + str(iterCount) + " Begins")

    wordVectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{1,}',
        stop_words='english',
        ngram_range=(1, 1),
        max_features=t1
    )
    wordVectorizer.fit(allText)
    trainWordFeatures = wordVectorizer.transform(trainText)
    trainFeatures = hstack([trainWordFeatures, trainCharFeatures])
    cvScore = []
    for classy in classNames:
        trainTarget = train[classy]
        model = LogisticRegression(C = 0.1, solver='sag')
        cvScore.append(np.mean(cross_val_score(model, trainFeatures, trainTarget, scoring="roc_auc", cv=10)))
    cvScoreT1 = np.mean(cvScore)

    wordVectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{1,}',
        stop_words='english',
        ngram_range=(1, 1),
        max_features=t2
    )
    wordVectorizer.fit(allText)
    trainWordFeatures = wordVectorizer.transform(trainText)
    trainFeatures = hstack([trainWordFeatures, trainCharFeatures])
    cvScore = []
    for classy in classNames:
        trainTarget = train[classy]
        model = LogisticRegression(C=0.1, solver='sag')
        cvScore.append(np.mean(cross_val_score(model, trainFeatures, trainTarget, scoring="roc_auc", cv=10)))
    cvScoreT2 = np.mean(cvScore)

    if cvScoreT1 > cvScoreT2:
        hi = t2-1
        if cvScoreT1 > bestScore:
            bestScore = cvScoreT1
            bestW = t1
    else:
        lo = t1+1
        if cvScoreT2 > bestScore:
            bestScore = cvScoreT2
            bestW = t2
    print(bestW)
    print(bestScore)

#5182 was the best with the 10 fold cross-validation
