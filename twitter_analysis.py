#Carregar Tweets
import tweepy

#Criar DataFrame
import pandas as pd     
import numpy as np

#Manipulação de arquivo e objetos
import os
import pickle
from imageio import imread

#Treinamento
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

#Expressoes regulares
import re

#erros
import warnings

#Criar API
from flask import Flask, request
from flask_cors import CORS
import requests 
import json

#Criar nuvem de palavras
from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS

#Plotar graficos e imagens
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#Plotar Mapa
from geopy.geocoders import Nominatim
import folium
from folium import plugins
from geopy.geocoders import Nominatim

class TwitterAnalysis():
    
    def __init__(self, BASE_DADOS = None, QUERY_PARAM = None):
        self.api = self.auth()

        if(BASE_DADOS == None):
            self.BASE_DADOS = "baseDados.txt"
        if(QUERY_PARAM == None):
            self.QUERY_PARAM = "vacina"

        self.tweets = None
        self.info = None
        self.tweets_df = None

        self.base_path = 'Utilitarios/DataSet'
        self.train = []
        self.wordsPT = []
        self.wordsPT_sentiments = []

        self.file_stopwords = 'Utilitarios/stopwords.txt'
        self.stopwords = []
        self.words = None
        self.words_clean = None
        self.file_mask = 'Utilitarios/mask.png'

        self.image_word_cloud = 'Word_Cloud.png'
        self.image_graphic = 'Graphic.png'
        self.image_Time_line = 'Time_line.png'
        self.image_Heat_Map = 'Heat_Map.html'
    
        self.vectorizer = None
        self.modelo = None



    def auth(self): 


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
            
            # if(count == 10):
            #     break
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

    
    def trainning(self):
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

        pos_tweets = [ tweet for index, tweet in enumerate(self.tweets_df['Tweets']) if self.tweets_df['SA NLTK'][index] > 0]
        neg_tweets = [ tweet for index, tweet in enumerate(self.tweets_df['Tweets']) if self.tweets_df['SA NLTK'][index] < 0]

        return pos_tweets, neg_tweets

    def get_result(self,positive,negative):
        return len(positive)*100/len(self.tweets_df['Tweets']), len(negative)*100/len(self.tweets_df['Tweets'])

    def create_json(self):
        json_tweets = self.tweets_df.to_json(orient = 'records')
        return json.loads(json_tweets)

    def create_word_cloud(self):
        
        with open(self.file_stopwords, 'r', encoding = 'utf-8') as f:
            [self.stopwords.append(self.wordsPT) for line in f for self.wordsPT in line.split()]

        self.words = ' '.join(self.tweets_df['Tweets'])

        self.words_clean = " ".join([word for word in self.words.split()
                                    if 'https' not in word
                                        and not word.startswith('@')
                                        and word != 'RT'
                                    ])

        warnings.simplefilter('ignore')

        twitter_mask = imread(self.file_mask)

        wc = WordCloud(min_font_size = 10, 
                      max_font_size = 300, 
                      background_color = 'white', 
                      mode = "RGB",
                      stopwords = self.stopwords,
                      width = 2000,
                      height = 1000,
                      mask = twitter_mask,
                      normalize_plurals= True).generate(self.words_clean)

        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.savefig(self.image_word_cloud, dpi=300)
        plt.close()

        return self.image_word_cloud

    def create_graphic(self, percent_positive, percent_negative):
        sentiments = ['Positivos', 'Negativos']
        percents = [percent_positive, percent_negative]
    
        cores = ['green', 'red']
        explode = (0.1, 0) 
        total = sum(percents)

        plt.pie(percents, labels=sentiments, colors=cores, explode=explode, autopct=lambda p: '{:.2f}%'.format(p), shadow=True, startangle=90)
        plt.savefig(self.image_graphic, format='png')
        plt.close()

        return self.image_graphic

    def create_time_line(self):
        data = self.tweets_df

        data['Date'] = pd.to_datetime(data['Date']).apply(lambda x: x.date())

        tlen = pd.Series(data['Date'].value_counts(), index=data['Date'])
        tlen = sorted(tlen.items())
        
        x = [value[0].strftime("%d/%m/%y") for value in tlen]
        y = [value[1] for value in tlen]
        
        plt.figure(figsize=(10,6))
        plt.plot(x, y, color='green')
        plt.scatter(x, y, color='red')
        plt.savefig(self.image_Time_line, format='png')

        return self.image_Time_line
    
    def create_heat_map(self):
        geolocator = Nominatim(user_agent="TweeterSentiments")

        latitude = []
        longitude = []

        for user_location in self.tweets_df['User Location']:
            try:
                location = geolocator.geocode(user_location)
                latitude.append(location.latitude)
                longitude.append(location.longitude)
            except:
                continue

        coordenadas = np.column_stack((latitude, longitude))

        mapa = folium.Map(location=[-15.788497,-47.879873],zoom_start=4.)

        mapa.add_child(plugins.HeatMap(coordenadas))
        mapa.save(self.image_Heat_Map)

        return self.image_Heat_Map

app = Flask("twitter_analysis")
CORS(app)

@app.route("/execute", methods = ['POST'])
def execute_analysis():
    analise = TwitterAnalysis()
    analise.QUERY_PARAM = request.form['query_param']
    analise.tweets, analise.info = analise.search_tweets()
    analise.tweets_df =  analise.create_dataframe()
    analise.source_tweets(analise.tweets_df)
    analise.trainning()

    resultado = analise.execute()

    percent_positive, percent_negative =  analise.get_result(resultado[0],resultado[1])

    return {
        "data": analise.create_json(),
        "img_wd_path": analise.create_word_cloud(),
        "img_gp_path": analise.create_graphic(percent_positive,percent_negative),
        "img_tl_path": analise.create_time_line(),
        "html_hm_path": analise.create_heat_map()
    }

app.run(debug=True)