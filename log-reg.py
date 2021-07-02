# load packages
from utils import *

import nltk                                # Python library for NLP
from nltk.corpus import twitter_samples    # sample Twitter dataset from NLTK
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression

# import data
try: 
    nltk.data.find('corpora/twitter_samples')
except:
    nltk.download('twitter_samples')

# download the stopwords from NLTK
try: 
    nltk.data.find('stopwords/english')
except:
    nltk.download('stopwords')

# select the set of positive and negative tweets
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

#load data to variables
tweets = all_positive_tweets + all_negative_tweets ## Concatenate the lists. 
labels = np.append(np.ones((len(all_positive_tweets),1)), np.zeros((len(all_negative_tweets),1)), axis = 0)
train_x, test_x, train_y, test_y = train_test_split(tweets, labels, test_size=0.2, random_state=123)

# calculate frequencies
freqs = build_freqs(train_x, train_y)

# create input variables
train_X = np.zeros((len(train_x), 3))
train_X[:,0] = 1

for i, tweet in enumerate(train_x):
    p_tweet = process_tweet(tweet)
    train_X[i, 1:] = list(extract_features(p_tweet, freqs))

test_X = np.zeros((len(test_x), 3))
test_X[:,0] = 1

for i, tweet in enumerate(test_x):
    p_tweet = process_tweet(tweet)
    test_X[i, 1:] = list(extract_features(p_tweet, freqs))

# run logistic regression model
clf = LogisticRegression(random_state=123).fit(train_X, train_y.squeeze())

# calculate metrics
metrics = precision_recall_fscore_support(test_y, clf.predict(test_X), average="weighted")

# present matrices
print(f"""
Precision:\t{metrics[0]:.2f}
Recall:\t\t{metrics[1]:.2f}
F-1:\t\t{metrics[2]:.2f}
""")