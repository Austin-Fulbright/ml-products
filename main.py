# Import the necessary libraries
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from gensim.models import Word2Vec
from gensim import corpora
from gensim.models import LdaModel
import preprocess_nltk

# Define file path
file_path = '../data/marketing_sample_for_amazon_com-ecommerce__20200101_20200131__10k_data.csv'

# Load data and drop missing values
df = pd.read_csv(file_path)
df = df.dropna(subset=['About Product', 'Selling Price', 'Product Name', 'Category'])

# Apply text preprocessing
df['About Product'] = df['About Product'].apply(preprocess_nltk.preprocess_text)
df['About Product'] = df['About Product'].apply(preprocess_nltk.remove_phrase)
# Rename columns
df['Product Name'] = df['Product Name'].apply(preprocess_nltk.preprocess_text)
df.rename(columns={'Uniq Id':'Id', 'Shipping Weight':'Shipping Weight(Pounds)', 'Selling Price':'Selling Price($)'}, inplace=True)
# Using + operator
df['About Product'] = df['Product Name'] + " " + df['About Product'] + " " + df['Category']

# Clean 'Selling Price($)' column
df['Selling Price($)'] = df['Selling Price($)'].str.replace('$', '').str.replace(' ', '').str.split('.').str[0] + '.'
df = df[~df['Selling Price($)'].str.contains('[a-zA-Z]', na=False)]
df['Selling Price($)'] = df['Selling Price($)'].str.replace(',', '').astype(float)
df['Selling Price($)'] = df['Selling Price($)'].apply(lambda x: "{:.2f}".format(x)).astype(float)

# Calculate IQR
Q1 = df['Selling Price($)'].quantile(0.25)
Q3 = df['Selling Price($)'].quantile(0.75)
IQR = Q3 - Q1

# Define limits
lower_limit = Q1 - 1.5*IQR
upper_limit = Q3 + 1.5*IQR

# Filter outliers
df = df[(df['Selling Price($)'] >= lower_limit) & (df['Selling Price($)'] <= upper_limit)]


# Split item descriptions into words
item_descriptions = df['About Product']
sentences = [desc.split() for desc in item_descriptions]

# Train Word2Vec model
word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# Generate word embeddings
X_word_embeddings = []
for desc in item_descriptions:
    word_vectors = [word2vec_model.wv[word] for word in desc.split() if word in word2vec_model.wv]
    if word_vectors:
        X_word_embeddings.append(np.mean(word_vectors, axis=0))
    else:
        X_word_embeddings.append(np.zeros(word2vec_model.vector_size))

X_word_embeddings = np.array(X_word_embeddings)

# Generate LDA topic distributions
dictionary = corpora.Dictionary(sentences)
corpus = [dictionary.doc2bow(desc) for desc in sentences]
num_topics = 10
lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)
X_topicdistribution = np.array([[dict(lda_model[dictionary.doc2bow(desc.split())]).get(topic, 0) for topic in range(num_topics)] for desc in item_descriptions])

# Combine word embeddings and topic distributions
X_combined = np.concatenate((X_word_embeddings, X_topicdistribution), axis=1)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, df['Selling Price($)'], test_size=0.2, random_state=42)

# Train SVR model
model = SVR(kernel='rbf')
model.fit(X_train, y_train)

# Make predictions and evaluate model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print("Mean Squared Error:", mse)
print("R-squared:", r2)




