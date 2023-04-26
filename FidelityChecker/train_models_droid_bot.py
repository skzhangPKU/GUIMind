from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import FunctionTransformer
from utils.file_util import readPklFile
from utils.evaluation import Score
from warnings import simplefilter
from utils.data import split_set
import numpy as np
import datetime
simplefilter(action='ignore', category=FutureWarning)

def to_dense(x):
    return x.todense()

def train():
    # load dataset
    data, raw_labels = readPklFile('samples/label_dataset.pkl')
    db_process_raw_data, db_process_remove_data = split_set(data, raw_labels, 0.4, shuffle=True)
    db_data, db_raw_lables = zip(*db_process_raw_data)
    train_data, test_data = split_set(db_data,db_raw_lables,0.2,shuffle=True)
    training_samples_raw, training_labels_raw = zip(*train_data)
    test_samples_raw, test_labels_raw = zip(*test_data)
    print('training sample number is ',len(training_samples_raw))
    print('test sample number is ', len(test_samples_raw))
    # samples finished
    label_encoder = LabelEncoder().fit(training_labels_raw)
    training_labels = label_encoder.transform(training_labels_raw)
    # model
    clf = Pipeline([
        ("vectorizer", CountVectorizer(stop_words="english",
                                       ngram_range=(1, 2))),
        ("tfidf", TfidfTransformer()),
        ("to_dense", FunctionTransformer(to_dense,
                                         accept_sparse=True)),
        ("gbc", GradientBoostingClassifier(n_estimators=200, random_state=5000))
    ])
    clf.fit(training_samples_raw, training_labels)

    test_samples = test_samples_raw
    test_labels = np.array(test_labels_raw)
    test_prediction = clf.predict(test_samples)
    print(test_prediction)
    print(test_labels)
    scores = Score._calculate_scores(test_prediction,test_labels)
    Score._logging_scores(scores)

if __name__ == '__main__':
    start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(start_time)
    train()
    end_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(end_time)


# 2022-12-08 16:50:48
# training sample number is  1167
# test sample number is  292
# [1 1 1 0 1 1 1 1 1 1 1 0 1 1 1 1 1 0 1 1 1 1 0 1 0 1 1 1 1 1 1 1 1 0 0 1 1
#  1 1 1 0 1 1 0 0 1 1 1 1 1 1 0 1 1 1 1 0 0 1 1 1 0 1 1 1 1 0 1 1 1 1 1 1 1
#  1 1 1 0 1 0 1 1 0 1 0 1 1 0 1 1 0 1 1 1 1 1 0 1 1 0 1 1 1 1 1 1 1 1 1 1 0
#  1 1 0 1 1 1 1 1 1 0 1 1 1 1 0 1 1 1 1 1 0 1 1 1 1 1 1 1 1 0 0 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 0 1 1 1 0 1 0 1 1 1 1 0 1 1 1 1 0 1 1 1
#  1 0 1 1 1 1 1 1 0 1 1 0 1 1 0 1 1 1 0 1 0 1 0 1 1 1 1 1 1 1 0 1 0 1 1 0 1
#  0 1 0 1 0 1 0 1 1 0 1 1 0 1 1 1 1 1 0 1 1 0 1 1 0 1 0 0 1 1 1 1 1 1 1 1 1
#  0 0 0 1 0 1 1 1 0 1 0 1 1 1 1 0 1 1 1 0 1 1 1 1 1 1 1 1 0 1 0 1 1]
# [1 1 1 0 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 0 1 0 1 1 1 1 1 1 0 1 0 0 1 1
#  1 1 1 0 0 1 0 0 1 1 1 1 1 1 0 1 1 1 1 0 0 1 1 1 0 1 0 1 1 1 1 1 1 1 1 1 1
#  1 1 1 0 1 0 1 1 0 1 0 1 1 1 1 1 0 1 1 1 1 1 1 1 1 0 1 1 1 1 0 1 1 1 1 1 0
#  1 1 0 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 0 1 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 0 1 1 1 0 1 0 1 1 1 1 0 1 1 1 0 0 1 1 1
#  1 0 1 1 1 1 1 1 0 0 1 1 1 1 0 0 1 1 0 1 0 1 0 0 1 1 1 1 1 1 0 1 0 1 1 0 1
#  0 1 0 1 0 1 1 1 1 0 1 1 0 1 1 1 1 1 0 1 1 0 1 1 0 1 1 0 1 1 1 1 0 0 1 1 1
#  0 0 0 0 0 1 1 1 0 1 0 1 1 1 1 0 1 1 1 0 1 1 1 1 1 1 1 1 0 0 0 1 1]
# 2022-12-08 17:03:20.030 | INFO     | utils.evaluation:_logging_scores:44 - [Result] TP: 211, FN: 8, FP: 14, TN: 59
# 2022-12-08 17:03:20.030 | INFO     | utils.evaluation:_logging_scores:45 - [Accuracy]: 0.9246575342465754
# 2022-12-08 17:03:20.030 | INFO     | utils.evaluation:_logging_scores:46 - [Precision]: 0.9377777777777778
# 2022-12-08 17:03:20.030 | INFO     | utils.evaluation:_logging_scores:47 - [Recall]: 0.9634703196347032
# 2022-12-08 17:03:20.030 | INFO     | utils.evaluation:_logging_scores:48 - [Score]: 0.9504504504504505
# 2022-12-08 17:03:20
