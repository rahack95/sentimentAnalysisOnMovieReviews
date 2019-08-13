from os.path import expanduser
from nltk.tag.stanford import StanfordPOSTagger
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import wordnet as wn
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import VotingClassifier


sklearn_tfidf = TfidfVectorizer(min_df=5, max_df = 0.8, sublinear_tf=True, use_idf=True)


import nltk
import random
import pickle
import threading
import re
import os
import datetime

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

pos_directory = 'short_reviews/train/pos'
neg_directory = 'short_reviews/train/neg'
'''
i = 0
for filename in os.listdir(pos_directory):
    if i < 5000:
        if filename.endswith(".txt"):
            short_pos += open(os.path.join(pos_directory, filename), encoding="ISO-8859-1").readlines()
        else:
            continue
    i += 1
i = 0
for filename in os.listdir(neg_directory):
    if i < 5000:
        if filename.endswith(".txt"):
            short_neg += open(os.path.join(neg_directory, filename), encoding="ISO-8859-1").readlines()
        else:
            continue
    i += 1
'''
# move this up here
all_words = []
documents = []
all_token_processed_words = []
all_token_processed_words_pos = []
all_token_processed_words_neg = []
all_labels = []
all_labels_pos = []
all_labels_neg = []
all_labels_test = []

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
        chunk_size = 10

    avg = len(review_sentences) / float(chunk_size)
    out = []
    last = 0.0
    while last < len(review_sentences):
        out.append(review_sentences[int(last):int(last + avg)])
        last += avg
    return out

def negated_words(word_tokens):
    negate_list = []
    modifier = None
    negative_territory = 0
    for j in range(len(word_tokens)):
        word = word_tokens[j]
        neg_verbs = ["n't"]
        for i in neg_verbs:
            if i in word:
                modifier = "vrbAdj"
                negative_territory = 4
        neg_verbs = ["not", "hardly"]
        if word in neg_verbs:
            modifier = "vrbAdj"
            negative_territory = 4
        neg_nouns = ["no", "none"]
        if word in neg_nouns:
            modifier = "nouns"
            negative_territory = 4
        if negative_territory > 0:
            pos = nltk.pos_tag([word])
            pos = pos[0][1]
            if (re.match('VB[G,P,D]*', pos) or re.match(('JJ|RB'), pos)) and modifier == "vrbAdj":
                if word not in stop_words_set: negate_list.append(j)
            elif re.match('NN.*', pos) and modifier == "nouns":
                if word not in stop_words_set: negate_list.append(j)
            negative_territory -= 1
    return negate_list

# Parse the positive reviews, do POS Tagging and extract adjectives
def parse_pos_reviews(short_pos_chunk):
    for p in short_pos_chunk:
        documents.append((p, "pos"))
        words = word_tokenize(p)
        pos = nltk.pos_tag(words)
        negate_list = set(negated_words(words))
        tokens_in_this_doc = []
        j = 0
        for w in pos:
            if w[1][0] in allowed_word_types:
                if w[0] not in stop_words_set:
                    if j in negate_list:
                        all_words.append("not_"+(stemmer.stem(w[0].lower())))
                        tokens_in_this_doc.append("not_" + (stemmer.stem(w[0].lower())))
                    else:
                        all_words.append(stemmer.stem(w[0].lower()))
                        tokens_in_this_doc.append(stemmer.stem(w[0].lower()))
            j += 1
        all_token_processed_words_pos.append(" ".join(tokens_in_this_doc))
        all_labels_pos.append(1)

# Parse the negative reviews, do POS Tagging and extract adjectives
def parse_neg_reviews(short_neg_chunk):
    for p in short_neg_chunk:
        documents.append((p, "neg"))
        words = word_tokenize(p)
        pos = nltk.pos_tag(words)
        negate_list = set(negated_words(words))
        tokens_in_this_doc = []
        j = 0
        for w in pos:
            if w[1][0] in allowed_word_types:
                if w[0] not in stop_words_set:
                    if j in negate_list:
                        all_words.append("not_" + (stemmer.stem(w[0].lower())))
                        tokens_in_this_doc.append("not_" + (stemmer.stem(w[0].lower())))
                    else:
                        all_words.append(stemmer.stem(w[0].lower()))
                        tokens_in_this_doc.append(stemmer.stem(w[0].lower()))
            j += 1
        all_token_processed_words_neg.append(" ".join(tokens_in_this_doc))
        all_labels_neg.append(0)

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
word_features = list(all_words.keys())[:6000]

all_token_processed_words = all_token_processed_words_pos[:4265]
all_token_processed_words = all_token_processed_words + all_token_processed_words_neg[:4265]
all_labels = all_labels_pos[:4265]
all_labels = all_labels + all_labels_neg[:4265]

all_token_processed_words_test = all_token_processed_words_pos[4265:]
all_token_processed_words_test = all_token_processed_words_test + all_token_processed_words_neg[4265:]
all_labels_test = all_labels_pos[4265:]
all_labels_test = all_labels_test + all_labels_neg[4265:]
all_labels_test = np.array(all_labels_test)

sklearn_representation_train = sklearn_tfidf.fit_transform(all_token_processed_words)
sklearn_representation_test = sklearn_tfidf.transform(all_token_processed_words_test)


GNB_classifier = GaussianNB()

#GNB_classifier.fit(sklearn_representation_train,np.array(all_labels))
#print("Original Naive Bayes Algo accuracy percent:", (accuracy_score(GNB_classifier.predit(sklearn_representation_test), all_labels_test)) * 100)

###############
#save_classifier = open("pickled_algos/originalnaivebayes5k_tf_idf.pickle", "wb")
#pickle.dump(GNB_classifier, save_classifier)
#save_classifier.close()

utc_timestamp = datetime.datetime.utcnow()
MNB_classifier = MultinomialNB()
MNB_classifier.fit(sklearn_representation_train,np.array(all_labels))
print("MNB_classifier accuracy percent:", (accuracy_score(MNB_classifier.predict(sklearn_representation_test), all_labels_test)) * 100)
time_taken_to_train = -1* (utc_timestamp - datetime.datetime.utcnow()).total_seconds()
print("Time taken to train: "+str(time_taken_to_train))
print(confusion_matrix(all_labels_test, MNB_classifier.predict(sklearn_representation_test)))

save_classifier = open("pickled_algos/MNB_classifier5k_tf_idf.pickle", "wb")
pickle.dump(MNB_classifier, save_classifier)
save_classifier.close()

utc_timestamp = datetime.datetime.utcnow()
BernoulliNB_classifier =BernoulliNB()
BernoulliNB_classifier.fit(sklearn_representation_train,np.array(all_labels))
print("BernoulliNB_classifier accuracy percent:", (accuracy_score(BernoulliNB_classifier.predict(sklearn_representation_test), all_labels_test)) * 100)
time_taken_to_train = -1* (utc_timestamp - datetime.datetime.utcnow()).total_seconds()
print("Time taken to train: "+str(time_taken_to_train))
print(confusion_matrix(all_labels_test, BernoulliNB_classifier.predict(sklearn_representation_test)))
save_classifier = open("pickled_algos/BernoulliNB_classifier5k_tf_idf.pickle", "wb")
pickle.dump(BernoulliNB_classifier, save_classifier)
save_classifier.close()

utc_timestamp = datetime.datetime.utcnow()
LogisticRegression_classifier = LogisticRegression()
LogisticRegression_classifier.fit(sklearn_representation_train,np.array(all_labels))
print("LogisticRegression_classifier accuracy percent:",
      (accuracy_score(LogisticRegression_classifier.predict(sklearn_representation_test), all_labels_test)) * 100)
time_taken_to_train = -1* (utc_timestamp - datetime.datetime.utcnow()).total_seconds()
print("Time taken to train: "+str(time_taken_to_train))
print(confusion_matrix(all_labels_test, LogisticRegression_classifier.predict(sklearn_representation_test)))
save_classifier = open("pickled_algos/LogisticRegression_classifier5k_tf_idf.pickle", "wb")
pickle.dump(LogisticRegression_classifier, save_classifier)
save_classifier.close()

utc_timestamp = datetime.datetime.utcnow()
LinearSVC_classifier = LinearSVC()
LinearSVC_classifier.fit(sklearn_representation_train,np.array(all_labels))
print("LinearSVC_classifier accuracy percent:", (accuracy_score(LinearSVC_classifier.predict(sklearn_representation_test), all_labels_test)) * 100)
time_taken_to_train = -1* (utc_timestamp - datetime.datetime.utcnow()).total_seconds()
print("Time taken to train: "+str(time_taken_to_train))
print(confusion_matrix(all_labels_test, LinearSVC_classifier.predict(sklearn_representation_test)))
save_classifier = open("pickled_algos/LinearSVC_classifier5k_tf_idf.pickle", "wb")
pickle.dump(LinearSVC_classifier, save_classifier)
save_classifier.close()

utc_timestamp = datetime.datetime.utcnow()
NuSVC_classifier = NuSVC()
NuSVC_classifier.fit(sklearn_representation_train,np.array(all_labels))
print("NuSVC_classifier accuracy percent:", (accuracy_score(NuSVC_classifier.predict(sklearn_representation_test), all_labels_test))*100)
time_taken_to_train = -1* (utc_timestamp - datetime.datetime.utcnow()).total_seconds()
print("Time taken to train: "+str(time_taken_to_train))
print(confusion_matrix(all_labels_test, NuSVC_classifier.predict(sklearn_representation_test)))
save_classifier = open("pickled_algos/NuSVC_classifier5k_tf_idf.pickle", "wb")
pickle.dump(NuSVC_classifier, save_classifier)
save_classifier.close()

utc_timestamp = datetime.datetime.utcnow()
SGDC_classifier = SGDClassifier()
SGDC_classifier.fit(sklearn_representation_train,np.array(all_labels))
print("SGDClassifier accuracy percent:", accuracy_score(SGDC_classifier.predict(sklearn_representation_test), all_labels_test) * 100)
time_taken_to_train = -1* (utc_timestamp - datetime.datetime.utcnow()).total_seconds()
print("Time taken to train: "+str(time_taken_to_train))
print(confusion_matrix(all_labels_test, SGDC_classifier.predict(sklearn_representation_test)))
save_classifier = open("pickled_algos/SGDC_classifier5k_tf_idf.pickle", "wb")
pickle.dump(SGDC_classifier, save_classifier)
save_classifier.close()

utc_timestamp = datetime.datetime.utcnow()
RF_classifier = RandomForestClassifier()
RF_classifier.fit(sklearn_representation_train,np.array(all_labels))
print("RFClassifier accuracy percent:", accuracy_score(RF_classifier.predict(sklearn_representation_test), all_labels_test) * 100)
time_taken_to_train = -1* (utc_timestamp - datetime.datetime.utcnow()).total_seconds()
print("Time taken to train: "+str(time_taken_to_train))
print(confusion_matrix(all_labels_test, RF_classifier.predict(sklearn_representation_test)))
save_classifier = open("pickled_algos/RF_classifier5k_tf_idf.pickle", "wb")
pickle.dump(RF_classifier, save_classifier)
save_classifier.close()

utc_timestamp = datetime.datetime.utcnow()
AB_classifier = AdaBoostClassifier(DecisionTreeClassifier(),algorithm="SAMME", n_estimators=200)
AB_classifier.fit(sklearn_representation_train,np.array(all_labels))
print("ABClassifier accuracy percent:", accuracy_score(AB_classifier.predict(sklearn_representation_test), all_labels_test) * 100)
time_taken_to_train = -1* (utc_timestamp - datetime.datetime.utcnow()).total_seconds()
print("Time taken to train: "+str(time_taken_to_train))
print(confusion_matrix(all_labels_test, AB_classifier.predict(sklearn_representation_test)))
save_classifier = open("pickled_algos/AB_classifier5k_tf_idf.pickle", "wb")
pickle.dump(AB_classifier, save_classifier)
save_classifier.close()

utc_timestamp = datetime.datetime.utcnow()
eclf1 = VotingClassifier(estimators=[('MNB', MNB_classifier), ('BNB', BernoulliNB_classifier),
                                     ('LR', LogisticRegression_classifier),
                                     ('LSVC',LinearSVC_classifier),('NSVC',NuSVC_classifier),('SGDC',SGDC_classifier),
                                     ('RF',RF_classifier), ('AB',AB_classifier)
                                    ], voting='hard'
                         )
eclf1.fit(sklearn_representation_train,np.array(all_labels))
print("Voting Classifier accuracy percent:", accuracy_score(eclf1.predict(sklearn_representation_test), all_labels_test) * 100)
time_taken_to_train = -1* (utc_timestamp - datetime.datetime.utcnow()).total_seconds()
print("Time taken to train: "+str(time_taken_to_train))
print(confusion_matrix(all_labels_test, eclf1.predict(sklearn_representation_test)))