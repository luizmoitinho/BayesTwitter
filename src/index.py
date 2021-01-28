from controllers.TwitterController import TwitterController
from persistencia.File import File
from persistencia.Connection import Connection

twitter = TwitterController()
_file = File('base_tweets')
twitter.get_result()

#data = _file.load_log()
#print(data['covid-19'][0]['created_at'])
#twitter = TwitterController()
#twitter.get_result()#