######################################
### IMPORTING PACKAGES AND SETTINGS###

# DataFrame
import pandas as pd

# Keras
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Spacy Tokenizer
from spacy.tokenizer import Tokenizer

# nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.util import ngrams

# Utility
import re
import pickle
import spacy
import en_core_web_sm
import string
from itertools import chain

# TEXT CLENAING
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

# SENTIMENT
POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"
SENTIMENT_THRESHOLDS = (0.4, 0.7)

# EXPORT
KERAS_MODEL = "model.h5"
WORD2VEC_MODEL = "w2v_model.w2v"
WORD2VEC_PICKLE = "w2v_model.pkl"
TOKENIZER_MODEL = "tokenizer.pkl"
ENCODER_MODEL = "encoder.pkl"

# STOPWORDS
stop_words = stopwords.words("english")
stemmer = SnowballStemmer("english")

### END OF IMPORTING PACKAGES AND SETTINGS ###
##############################################


##############################################
### DEFINING FUNCTIONS FOR USER EVALUATION ###

# Main function to predict sentiment and keyword detection
def predict(text, include_neutral=True):
    new_list = []
    intersect_key = []
    if not intersect_key:
        text = text.translate(str.maketrans('', '', string.punctuation))
        n_grams = ngrams(text.split(), 2)
        Separate = ' '
        for grams in n_grams:
            new_list.append(Separate.join(grams))
        intersect_key.append(
            list(set(new_list).intersection(suicide_dictionary)))

        tokenizer_doc = custom_nlp(text)
        tokens_doc = [token.text for token in tokenizer_doc]
        tokens_doc = list(dict.fromkeys(tokens_doc))
        intersect_key.append(
            list(set(tokens_doc).intersection(suicide_dictionary)))

    if not intersect_key:
        new_lst = []
        for x in tokens_doc:
            new_lst.extend(w2v_model.most_similar(x))

        new_lst = [x[0] for x in new_lst]
        tokens_doc.append(new_lst)
        intersect_key.append(
            list(set(tokens_doc).intersection(suicide_dictionary)))

    # Tokenize text
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=300)
    # Predict
    score = model.predict([x_test])[0]
    # Decode sentiment
    label = decode_sentiment(score, include_neutral=include_neutral)
    keywords = list(chain.from_iterable(intersect_key))
    keywords_str = ','.join([str(i) for i in keywords])
    return label, float(score), keywords_str

# Function to assign label to the score of the sentiment analysis model
def decode_sentiment(score, include_neutral=True):
    if include_neutral:
        label = NEUTRAL
        if score <= 0.4:
            label = NEGATIVE
        elif score >= 0.7:
            label = POSITIVE

        return label
    else:
        return NEGATIVE if score < 0.5 else POSITIVE


# Function to Clean, remove stopwords and unwanted characters from the input text
def preprocess(text, stem=False):
    # Remove link,user and special characters
    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)

def user_evaluation(sentiment, score, keywords_str):
    if sentiment == "NEGATIVE":
        print("Depression detected")
        print("Intensity: %.02f" % ((1-score)*100))
        print("Keywords Detected:", keywords_str)
    else:
        print("Patient is in good Mental Health")
        print("Sentiment:", sentiment)
        print("Intensity: %f" % (score*100))

######################
### LOADING MODELS ###

# Loading Tokenizer
with open(TOKENIZER_MODEL, 'rb') as handle:
    tokenizer = pickle.load(handle)
# Loading Encoder    
with open(ENCODER_MODEL, 'rb') as handle:
    encoder = pickle.load(handle)

# Loading Keras Sequential Model
model = keras.models.load_model(KERAS_MODEL)

# Loading Word2Vector Model
#w2v_model = gensim.models.word2vec.Word2Vec.load('model.w2v')
with open(WORD2VEC_PICKLE, 'rb') as handle:
    w2v_model = pickle.load(handle)


# Loading Keywords for detection
dictionary = pd.read_excel("Input/keyword_dictionary.xlsx")
suicide_dictionary = dictionary['keywords'].tolist()

# Tokenizer for keyword detection
custom_nlp = en_core_web_sm.load()
prefix_re = spacy.util.compile_prefix_regex(custom_nlp.Defaults.prefixes)
suffix_re = spacy.util.compile_suffix_regex(custom_nlp.Defaults.suffixes)
infix_re = re.compile(r'''[-~]''')
def customize_tokenizer(nlp):
    # Adds support to use `-` as the delimiter for tokenization
    return Tokenizer(nlp.vocab, prefix_search=prefix_re.search,
                     suffix_search=suffix_re.search,
                     infix_finditer=infix_re.finditer,
                     token_match=None
                     )

custom_nlp.tokenizer = customize_tokenizer(custom_nlp)

### END OF LOADING MODELS ###
#############################