import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

new_york_tweets = pd.read_json('new_york.json', lines=True)
london_tweets = pd.read_json('london.json', lines=True)
paris_tweets = pd.read_json('paris.json', lines=True)

new_york_text = new_york_tweets['text'].tolist()
london_text = london_tweets['text'].tolist()
paris_text = paris_tweets['text'].tolist()

# combine all text into one long list of tweets
all_tweets = new_york_text + london_text + paris_text

# create labels for tweets by location; 0 = new york, 1 = london, 2 = paris
labels = [0] * len(new_york_text) + [1] * len(london_text) + [2] * len(paris_text)

# divide set into train and test set
train_data, test_data, train_labels, test_labels = train_test_split(all_tweets, labels, test_size=0.2, random_state=1)

# create a counter and transform train/test data
counter = CountVectorizer()
counter.fit(train_data + test_data)
train_counts = counter.transform(train_data)
test_counts = counter.transform(test_data)

# create a NB classifier, fit it with the training data and create predictions on the test data to evaluate the model
classifier = MultinomialNB()
classifier.fit(train_counts, train_labels)
predictions = classifier.predict(test_counts)

# classify by using accuracy_score
# print(accuracy_score(test_labels, predictions))

# classify by using confusion matrix
# print(confusion_matrix(test_labels, predictions))
