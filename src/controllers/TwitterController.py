from persistencia.Connection import Connection
from TwitterSearch import *
from textblob import TextBlob as tb
from persistencia.File import File
import tweepy
import numpy as np
import datetime
import json


class TwitterController:
    
    def __init__(self):
        connection = Connection()
        self.connection = connection.fn_getToken()
 
    def get_result(self):
        try:
            ts = TwitterSearch(
                consumer_key = self.connection['consumer_key'],
                consumer_secret = self.connection['consumer_secret'],
                access_token = self.connection['access_token'],
                access_token_secret = self.connection['access_token_secret']
            )

            tso = TwitterSearchOrder()
            keywords = 'leite condensado'
            tso.set_keywords([keywords])

            tso.set_since(datetime.date(2020, 2, 9))
            tso.set_language('pt')
   
            #tso.set_geocode(-10.6470236,-39.9841415,100)

            i = 0             
            file = File('base_tweets.json',[])
            jsonData = []
            for tweet in ts.search_tweets_iterable(tso):
                if(i >= 1000 ):
                    break
                jsonData.append({
                    "created_at": tweet['created_at'],
                    "id": tweet['id'],
                    "text": tweet['text'],
                    "entities": {
                        "hashtags": tweet['entities']['hashtags'],
                        "symbols": tweet['entities']['symbols'],
                    },
                    "user": {
                        "location":  tweet['user']['location'],
                        "created_at":  tweet['user']['created_at'],
                        "favourites_count": tweet['user']['favourites_count'],
                    },
                    "geo": tweet['geo'],
                    "coordinates": tweet['coordinates'],
                    "place": tweet['place'],
                })

                i = i+1

            file.save(keywords,jsonData)

        except TwitterSearchException as e:
            print(e)
