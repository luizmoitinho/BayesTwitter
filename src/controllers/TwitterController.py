from persistencia.Connection import fn_getToken
from TwitterSearch import *
from textblob import TextBlob as tb
import tweepy
import numpy as np


class TwitterController:
    
    def __init__(self):
       self.connection = fn_getToken()
 
    def get_result(self):
        try:
            ts = TwitterSearch(
                self.connection
            )

            tso = TwitterSearchOrder()
            tso.set_keywords(['covid-19','covid','pandemia',
                              'vacina','saude','saúde']
                            )
            tso.set_language('pt')
            #tso.set_geocode(-10.6470236,-39.9841415,100)

            
            i = 0  
            metrica = []             
            for tweet in ts.search_tweets_iterable(tso):
                print(i) 
                i = i+1
                print( '@%s tweeted: %s' % ( tweet['user']['screen_name'], tweet['text'] ) )
                analysis = tb(tweet['text'])
                polarity = analysis.sentiment.polarity
                metrica.append(polarity)
                print(i+1,tweet['text'],polarity,"\n")

            print('MÉDIA DE SENTIMENTO: ' + str(np.mean(metrica)))

        except TwitterSearchException as e:
            print(e)
