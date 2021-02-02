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


class TwitterAnalysis():
    
    def __init__(self):
        self.BASE_DADOS = "baseDados.txt"
        self.QUERY_PARAM = "assassino"
        self.api = self.auth()
        
        self.tweets = None
        self.info = None
        self.tweets_df = None

        self.base_path = 'DataSet'
        self.train = []
        self.wordsPT = []
        self.wordsPT_sentiments = []
    
        self.vectorizer = None
        self.modelo = None

    def auth(self):
        consumer_key='srt5WEhYgrJ5SFGkCtkDSvjzH'
        consumer_secret='HF91PBFEO7PyKBOUj1JRsFUutHW0VrXstHSxzISR6mkU32eHN1'
        access_token='3314192050-3FzJ7jYYGnkLxm5DUarXOXUEgvIsgQhvgvEKraJ'
        access_token_secret='CASsGqefKg7SFlzf3K7b7LvrlC7jetxzIIAesBOxXdHuK'

        #Autentication Methods
        auth = tweepy.OAuthHandler(consumer_key,consumer_secret)
        auth.set_access_token(access_token,access_token_secret)
        return tweepy.API(auth)

    def search_tweets(self):
        tweets = []
        info = []
        count = 1
        for tweet in tweepy.Cursor(self.api.search,
                                q = self.QUERY_PARAM,
                                tweet_mode='extended',
                                rpp=100,
                                result_type="popular",
                                include_entities=True,
                                lang="pt").items(1500):

            if 'retweeted_status' in dir(tweet):
                aux = tweet.retweeted_status.full_text
            else:
                aux = tweet.full_text
                
            newtweet = aux.replace("\n", " ")
        
            tweets.append(newtweet)
            info.append(tweet)
            
            file = open(self.BASE_DADOS, "a", -1, "utf-8")
            file.write(newtweet+'\n')
            file.close()
            
            if(count == 50):
                break
            count = count + 1
        return tweets, info

    def create_dataframe(self):
        tweets_df = pd.DataFrame(self.tweets, columns=['Tweets']) 
        tweets_df['len']  = np.array([len(tweet) for tweet in self.tweets])
        tweets_df['ID']   = np.array([tweet.id for tweet in self.info])
        tweets_df['Date'] = np.array([tweet.created_at for tweet in self.info])
        tweets_df['Source'] = np.array([tweet.source for tweet in self.info])
        tweets_df['Likes']  = np.array([tweet.favorite_count for tweet in self.info])
        tweets_df['RTs']    = np.array([tweet.retweet_count for tweet in self.info])
        tweets_df['User Location']    = np.array([tweet.user.location for tweet in self.info])
        tweets_df['Geo']    = np.array([tweet.geo for tweet in self.info])
        tweets_df['Coordinates']    = np.array([tweet.coordinates for tweet in self.info])
        tweets_df.to_csv(self.BASE_DADOS)
        tweets_df.head()
        return tweets_df


    def source_tweets(self, tweets_df):
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
        return percent


    def clean_tweet(self, tweet):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

    
    def trainnig(self):
        files = [os.path.join(self.base_path, f) for f in os.listdir(self.base_path)]

        for file in files:
            t = 1 if '_positive' in file else -1
            with open(file, 'r', encoding='utf-8') as content_file:
                content = content_file.read()
                all = content.split('\n')
                for w in all:   
                    self.wordsPT.append(w)
                    self.wordsPT_sentiments.append(t)
                    self.train.append((w, t))


        self.vectorizer = CountVectorizer(analyzer="word")
        freq_tweets = self.vectorizer.fit_transform(self.wordsPT)
        self.modelo = MultinomialNB()
        return self.modelo.fit(freq_tweets, self.wordsPT_sentiments)

    def execute(self):
        tweetsarray = []

        for tw in self.tweets_df['Tweets']:
            text = self.clean_tweet(tw)
            tweetsarray.append(text)

        predictionData = self.vectorizer.transform(self.tweets_df['Tweets'])
        self.tweets_df['SA NLTK']  = self.modelo.predict(predictionData)

        for i in range(len(self.tweets_df['SA NLTK'])):
            print(self.tweets_df['Tweets'][i], ' : ', self.tweets_df['SA NLTK'][i])

        pos_tweets = [ tweet for index, tweet in enumerate(self.tweets_df['Tweets']) if self.tweets_df['SA NLTK'][index] > 0]
        neg_tweets = [ tweet for index, tweet in enumerate(self.tweets_df['Tweets']) if self.tweets_df['SA NLTK'][index] < 0]

        return pos_tweets, neg_tweets

    def display_result(self,positive,negative ):
        print("Porcentagem de Tweets Positivos: {}%".format(len(positive)*100/len(self.tweets_df['Tweets'])))
        print("Porcentagem de Tweets Negativos: {}%".format(len(negative)*100/len(self.tweets_df['Tweets'])))




analise = TwitterAnalysis()

analise.tweets, analise.info = analise.search_tweets()
analise.tweets_df =  analise.create_dataframe()
analise.source_tweets(analise.tweets_df)
analise.trainnig()
resultado = analise.execute()
analise.display_result(resultado[0],resultado[1])