import tweepy
import pandas as pd     
import numpy as np 
import time
import os
import re
import pickle
from textblob.classifiers import NaiveBayesClassifier
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

#-----------------TWITTER-----------------#

#Keys for autentication
consumer_key='srt5WEhYgrJ5SFGkCtkDSvjzH'
consumer_secret='HF91PBFEO7PyKBOUj1JRsFUutHW0VrXstHSxzISR6mkU32eHN1'

access_token='3314192050-3FzJ7jYYGnkLxm5DUarXOXUEgvIsgQhvgvEKraJ'
access_token_secret='CASsGqefKg7SFlzf3K7b7LvrlC7jetxzIIAesBOxXdHuK'

"""### Twitter Autentication"""

#Autentication Methods
auth = tweepy.OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_token_secret)
api = tweepy.API(auth)

"""### Searching for Tweets"""

#Searchin Twitter Timelines
tweets = []
info = []
count = 1
for tweet in tweepy.Cursor(api.search,
                           q="assassino",
                           tweet_mode='extended',
                           rpp=100,
                           result_type="popular",
                           include_entities=True,
                           lang="pt").items(1500):
    if 'retweeted_status' in dir(tweet):
        aux=tweet.retweeted_status.full_text
    else:
        aux=tweet.full_text
        
    newtweet = aux.replace("\n", " ")
   
    tweets.append(newtweet)
    info.append(tweet)
    
    file = open("baseDados.txt", "a", -1, "utf-8")
    file.write(newtweet+'\n')
    file.close()
    
    #time.sleep(0.5)
    if(count == 50):
        break
    count = count + 1

"""### Creating the dataframe """

#Construction of the dataframe
tweets_df = pd.DataFrame(tweets, columns=['Tweets']) 

tweets_df['len']  = np.array([len(tweet) for tweet in tweets])
tweets_df['ID']   = np.array([tweet.id for tweet in info])
tweets_df['Date'] = np.array([tweet.created_at for tweet in info])
tweets_df['Source'] = np.array([tweet.source for tweet in info])
tweets_df['Likes']  = np.array([tweet.favorite_count for tweet in info])
tweets_df['RTs']    = np.array([tweet.retweet_count for tweet in info])
tweets_df['User Location']    = np.array([tweet.user.location for tweet in info])
tweets_df['Geo']    = np.array([tweet.geo for tweet in info])
tweets_df['Coordinates']    = np.array([tweet.coordinates for tweet in info])

tweets_df.to_csv("baseDados.csv")

tweets_df.head()


"""### Source of Tweets"""

#Source of the Tweets
sources = []
for source in tweets_df['Source']:
    if source not in sources:
        sources.append(source)

percent = np.zeros(len(sources))

for source in tweets_df['Source']:
    for index in range(len(sources)):
        if source == sources[index]:
            percent[index] += 1
            pass


#------------------------------------------------------------#

base_path = 'DataSet'
train = []
wordsPT = []
wordsPT_sentiments = []

files = [os.path.join(base_path, f) for f in os.listdir(base_path)]

for file in files:
    t = 1 if '_positive' in file else -1
    with open(file, 'r', encoding='utf-8') as content_file:
        content = content_file.read()
        all = content.split('\n')
        for w in all:
            wordsPT.append(w)
            wordsPT_sentiments.append(t)
            train.append((w, t))

# wordsPT = []
# wordsPT_sentiments = []
# train = []

# #Importando o Léxico de Palavras com polaridades
# sentilexpt = open('SentiLex-lem-PT02.txt')

# #Criando um dicionário de palavras com a respectiva polaridade.
# for i in sentilexpt.readlines():
#     pos_ponto = i.find('.')
#     palavra = (i[:pos_ponto])
#     pol_pos = i.find('POL')
#     polaridade = (i[pol_pos+7:pol_pos+9]).replace(';', '')
#     wordsPT.append(palavra)
#     wordsPT_sentiments.append(int(polaridade))
#     train.append((palavra, int(polaridade)))

# cl = NaiveBayesClassifier(train)
# save_classifier = open("naivebayes.pickle","wb")
# pickle.dump(cl, save_classifier)
# save_classifier.close()
# classifier_f = open("naivebayes.pickle", "rb")
# classifier = pickle.load(classifier_f)
# classifier_f.close()

# cl = NaiveBayesClassifier(train)
# prob_dist = cl.prob_classify('Eu não me odeio')
# # print(prob_dist.max())
# print(prob_dist.prob(-1))
# print(prob_dist.prob(1))

# for i in range(len(test)):
#   prob_dist = classifier.prob_classify(test[i])
#   print(f'[{i}] - ', prob_dist.max())

vectorizer = CountVectorizer(analyzer="word")
freq_tweets = vectorizer.fit_transform(wordsPT)
modelo = MultinomialNB()
modelo.fit(freq_tweets,wordsPT_sentiments)

### Sentiment Analisys
def clean_tweet(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

tweetsarray = []

for tw in tweets_df['Tweets']:
    text = clean_tweet(tw)
    tweetsarray.append(text)

predictionData = vectorizer.transform(tweets_df['Tweets'])
tweets_df['SA NLTK']  = modelo.predict(predictionData)

for i in range(len(tweets_df['SA NLTK'])):
  print(tweets_df['Tweets'][i], ' : ', tweets_df['SA NLTK'][i])

pos_tweets = [ tweet for index, tweet in enumerate(tweets_df['Tweets']) if tweets_df['SA NLTK'][index] > 0]
neg_tweets = [ tweet for index, tweet in enumerate(tweets_df['Tweets']) if tweets_df['SA NLTK'][index] < 0]

print("Porcentagem de Tweets Positivos: {}%".format(len(pos_tweets)*100/len(tweets_df['Tweets'])))
print("Porcentagem de Tweets Negativos: {}%".format(len(neg_tweets)*100/len(tweets_df['Tweets'])))

sentiments = ['Positivos', 'Negativos']
percents = [len(pos_tweets), len(neg_tweets)]