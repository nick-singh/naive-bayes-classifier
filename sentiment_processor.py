import json
import csv
import os
import re
import cPickle
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup


# __file__ refers to the file settings.py 
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

tweets_data_path =  os.path.join(APP_ROOT, 'training.1600000.processed.noemoticon.csv')

data_set_features =  os.path.join(APP_ROOT, '1600000_features.json')

# processed_data = os.path.join(APP_ROOT, 'training.60000.processed.noemoticon.json')

twitter_data = csv.reader(open(tweets_data_path,'rb'))

twitter_data = map(tuple,twitter_data)


def get_words_in_tweets(tweets):
	all_words = []
	for(words,sentiment) in tweets:
		all_words.extend(words)
	return all_words

def get_word_features(wordlist):
	wordlist = nltk.FreqDist(wordlist)
	word_features = wordlist.keys()
	return word_features


def tweet_to_words( raw_tweet ):
    # Function to convert a raw tweet to a string of words
    # The input is a single string (a raw movie tweet), and 
    # the output is a single string (a preprocessed movie tweet)
    #
    # 1. Remove HTML
    tweet_text = BeautifulSoup(raw_tweet).get_text() 
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", tweet_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()
    #
    # Removing the word RT
    words = [x for x in words if x != "rt"]                               
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6.  return the result.
    return meaningful_words

# Function accepts list of words and clean each row
def clean_list_of_words(list_of_words):
    
    # Get the number of tweets based on the datagrame column size
    num_tweets = len(list_of_words)
    #
    # Initialize an empty list to hold the clean tweets  
    clean_train_words = []  
    #
    print "Cleaning and parsing the training set items...\n"
    
    for i in xrange( 0, num_tweets ):
        # If the index is evenly divisible by 1000, print a messagegoo
        if( (i+1)%1000 == 0 ):
            print "item %d of %d\n" % ( i+1, num_tweets )                                                                    
        clean_train_words.append((tweet_to_words(list_of_words[i][5]),list_of_words[i][0]))
    return clean_train_words

def extract_features(document):
    document_words = set(document)
    word_features = json.load(open(data_set_features,'r'))
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)        
    return features

def set_word_features(features):
	word_features = features


clean_data = clean_list_of_words(twitter_data)
print "Dumping clean data..."
json.dump(clean_data, open('1600000_clean_data_set.json', 'w'))
# print "Loading processes data..."
# clean_data = json.load(open(processed_data,'r'))

# print "Extracting Features..."
# word_features = get_word_features(get_words_in_tweets(clean_data))
# print "Writing features to file..." 
# json.dump(word_features, open('data-sets/500-data-sets/500_features.json', 'w'))

# word_features = json.load(open('data-sets/500-data-sets/500_features.json','r'))

print "Applying Features..."
training_set = nltk.classify.apply_features(extract_features, clean_data)

print "Classifying..."
classifier = nltk.NaiveBayesClassifier.train(training_set)

print "Saving model.."
cPickle.dump(classifier, open('1600000_classifier.pickle','wb'))

