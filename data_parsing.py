from os.path import expanduser
from nltk.tag.stanford import StanfordPOSTagger
# from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import BaggingClassifier
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import wordnet as wn

import nltk
import random
import pickle
import threading
import os

# Configuring Stanford POS tagger
home = expanduser("~")
path_to_model = home + '/PycharmProjects/SentimentAnalysis/stanford-postagger/models/english-bidirectional-distsim.tagger'
path_to_jar = home + '/PycharmProjects/SentimentAnalysis/stanford-postagger/stanford-postagger.jar'
tagger = StanfordPOSTagger(path_to_model, path_to_jar)
tagger.java_options = '-mx512m'          ### Setting higher memory limit for long sentences
stemmer = SnowballStemmer("english")
stop_words_set = set(stopwords.words('english'))

def similar_word_set(word_text):
    word_text_set = set()
    for ss in wn.synsets(word_text):
        if '.a.' in ss.name() or '.s.' in ss.name() or '.r.' in ss.name():
            for l_name in ss.lemma_names():
                word_text_set.add(stemmer.stem(l_name.lower()))
    return word_text_set

# Load the classified positive and negative review files
short_pos = open("short_reviews/train/positive.txt", encoding="ISO-8859-1").readlines()
short_neg = open("short_reviews/train/negative.txt", encoding="ISO-8859-1").readlines()
'''
pos_directory = 'short_reviews/train/pos'
neg_directory = 'short_reviews/train/neg'
for filename in os.listdir(pos_directory):
    if filename.endswith(".txt"):
        short_pos += open(os.path.join(pos_directory, filename), encoding="ISO-8859-1").readlines()
    else:
        continue

for filename in os.listdir(neg_directory):
    if filename.endswith(".txt"):
        short_neg += open(os.path.join(neg_directory, filename), encoding="ISO-8859-1").readlines()
    else:
        continue
'''
# move this up here
all_words = []
documents = []

#  j is adject, r is adverb, and v is verb
# allowed_word_types = ["J","R","V"]
allowed_word_types = ["J", "R", "V"]


def chunkIt(review_sentences):
    chunk_size = 1
    if len(review_sentences) == 1:
        chunk_size = 1
    elif len(review_sentences) < 5:
        chunk_size = 2
    elif len(review_sentences) < 10:
        chunk_size = 5
    elif len(review_sentences) < 20:
        chunk_size = 5
    else:
        chunk_size = 100

    avg = len(review_sentences) / float(chunk_size)
    out = []
    last = 0.0
    while last < len(review_sentences):
        out.append(review_sentences[int(last):int(last + avg)])
        last += avg
    return out

# Parse the positive reviews, do POS Tagging and extract adjectives
def parse_pos_reviews(short_pos_chunk):
    for p in short_pos_chunk:
        documents.append((p, "pos"))
        words = word_tokenize(p)
        pos = nltk.pos_tag(words)
        #pos = tagger.tag(words)
        for w in pos:
            if w[1][0] in allowed_word_types:
                if w[0] not in stop_words_set:
                    all_words.append(stemmer.stem(w[0].lower()))

# Parse the negative reviews, do POS Tagging and extract adjectives
def parse_neg_reviews(short_neg_chunk):
    for p in short_neg_chunk:
        documents.append((p, "neg"))
        words = word_tokenize(p)
        pos = nltk.pos_tag(words)
        #pos = tagger.tag(words)
        for w in pos:
            if w[1][0] in allowed_word_types:
                if w[0] not in stop_words_set:
                    all_words.append(stemmer.stem(w[0].lower()))

def multithread_parse():
    global short_pos
    global short_neg

    # positives
    short_pos_batch_groups = chunkIt(short_pos)
    num_pos_batch_groups = len(short_pos_batch_groups)
    print(num_pos_batch_groups)
    threads = []
    for i in range(num_pos_batch_groups):
        threads.append(threading.Thread(target=parse_pos_reviews, args=(short_pos_batch_groups.pop(),)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # negatives
    short_neg_batch_groups = chunkIt(short_neg)
    num_neg_batch_groups = len(short_neg_batch_groups)
    print(num_neg_batch_groups)
    threads = []
    for i in range(num_neg_batch_groups):
        threads.append(threading.Thread(target=parse_neg_reviews, args=(short_neg_batch_groups.pop(),)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()


multithread_parse()

# Save all the adjectives to a file
save_documents = open("pickled_algos/documents.pickle", "wb")
pickle.dump(documents, save_documents)
save_documents.close()

all_words = nltk.FreqDist(all_words)
print(len(list(all_words.keys())))
word_features = list(all_words.keys())[:6000]

save_word_features = open("pickled_algos/word_features6k.pickle", "wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()
print("saved word features!")

def find_features(document):
    words = word_tokenize(document)
    words_set = set()
    for word in words:
        if word not in stop_words_set:
            words_set.add(stemmer.stem(word.lower()))

    features = {}
    for w in word_features:
        features[w] = (w in words_set)

    return features


featuresets = [(find_features(rev), category) for (rev, category) in documents]

random.shuffle(featuresets)
print(len(featuresets))
testing_set = featuresets[10000:]
training_set = featuresets[:10000]

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

'''
classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set)) * 100)
classifier.show_most_informative_features(15)

###############
save_classifier = open("pickled_algos/originalnaivebayes5k.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set)) * 100)

save_classifier = open("pickled_algos/MNB_classifier5k.pickle", "wb")
pickle.dump(MNB_classifier, save_classifier)
save_classifier.close()

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set)) * 100)

save_classifier = open("pickled_algos/BernoulliNB_classifier5k.pickle", "wb")
pickle.dump(BernoulliNB_classifier, save_classifier)
save_classifier.close()

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:",
      (nltk.classify.accuracy(LogisticRegression_classifier, testing_set)) * 100)

save_classifier = open("pickled_algos/LogisticRegression_classifier5k.pickle", "wb")
pickle.dump(LogisticRegression_classifier, save_classifier)
save_classifier.close()

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set)) * 100)

save_classifier = open("pickled_algos/LinearSVC_classifier5k.pickle", "wb")
pickle.dump(LinearSVC_classifier, save_classifier)
save_classifier.close()

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)

save_classifier = open("pickled_algos/NuSVC_classifier5k.pickle", "wb")
pickle.dump(NuSVC_classifier, save_classifier)
save_classifier.close()


SGDC_classifier = SklearnClassifier(SGDClassifier())
SGDC_classifier.train(training_set)
print("SGDClassifier accuracy percent:", nltk.classify.accuracy(SGDC_classifier, testing_set) * 100)

save_classifier = open("pickled_algos/SGDC_classifier5k.pickle", "wb")
pickle.dump(SGDC_classifier, save_classifier)
save_classifier.close()

RF_classifier = RandomForestClassifier()
RF_classifier.train(training_set)
print("RFClassifier accuracy percent:", nltk.classify.accuracy(RF_classifier, testing_set) * 100)

save_classifier = open("pickled_algos/RF_classifier5k.pickle", "wb")
pickle.dump(RF_classifier, save_classifier)
save_classifier.close()
'''
GB_classifier = GradientBoostingClassifier()
GB_classifier.train(training_set)
print("GBClassifier accuracy percent:", nltk.classify.accuracy(GB_classifier, testing_set) * 100)

save_classifier = open("pickled_algos/GB_classifier5k.pickle", "wb")
pickle.dump(GB_classifier, save_classifier)
save_classifier.close()

voted_classifier = VoteClassifier(classifier,
                                  NuSVC_classifier,
                                  LinearSVC_classifier,
                                  SGDC_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier,
                                  GB_classifier,
                                  RF_classifier)

print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)

print("Classification:", voted_classifier.classify(testing_set[0][0]), "Confidence %:",voted_classifier.confidence(testing_set[0][0])*100)
print("Classification:", voted_classifier.classify(testing_set[1][0]), "Confidence %:",voted_classifier.confidence(testing_set[1][0])*100)
print("Classification:", voted_classifier.classify(testing_set[2][0]), "Confidence %:",voted_classifier.confidence(testing_set[2][0])*100)
print("Classification:", voted_classifier.classify(testing_set[3][0]), "Confidence %:",voted_classifier.confidence(testing_set[3][0])*100)
print("Classification:", voted_classifier.classify(testing_set[4][0]), "Confidence %:",voted_classifier.confidence(testing_set[4][0])*100)
print("Classification:", voted_classifier.classify(testing_set[5][0]), "Confidence %:",voted_classifier.confidence(testing_set[5][0])*100)
