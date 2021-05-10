import pandas as pd

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

