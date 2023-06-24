import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

def preprocess_text(text):
    
    tokens = nltk.word_tokenize(text.lower())


    tokens = [token for token in tokens if token not in string.punctuation]


    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]


    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]


    preprocessed_text = ' '.join(tokens)

    return preprocessed_text

def remove_phrase(text):
    # Replace the phrase with an empty string
    new_text = text.replace('make sure fit entering model number', '')
    return new_text

def remove_phrased(text):
    # Replace the phrase with an empty string
    new_text = text.replace('$', '')
    return new_text




