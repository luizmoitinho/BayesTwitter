import json
import time
import os 
from configparser import ConfigParser

class File:
    def __init__(self, nameFile, data):
        self.pathFile = os.path.dirname(os.path.realpath(__file__))+"/"+nameFile
        self.data = data

    def save(self, titulo = None, data = None):
        jsonData = self.load_log()

        if(titulo == None):
            titulo = time.strftime('%Y%m%d_%H:%M:%S', time.localtime())

        if(data != None ):
            self.data = data
        
        jsonData[titulo] = self.data
        with open(self.pathFile, 'w') as jsonFile:
            json.dump(jsonData, jsonFile, indent=2)

        # with open(pathFile, 'a') as file:
        #   file.write(',')

    def load(self):
        dataJson = {}
        if os.path.exists(self.pathFile):
            with open(self.pathFile,'r') as jsonFile:
                dataJson = json.load(jsonFile)
            
        return dataJson
    
    def load_log(self):
        dataJson = {}
        if os.path.exists(self.pathFile):
            with open(self.pathFile,'r') as jsonFile:
                dataJson = json.load(jsonFile)

        return dataJson


    def load_token(self):
        token = 'token'
        parser = ConfigParser()
        parser.read(self.pathFile)
        data = {}
        
        if parser.has_section(token):
            params = parser.items(token)
            for param in params:
                data[param[0]] = param[1]
        else:
            raise Exception('Section {0} not found in the {1} file'.format(token, self.pathFile))

        return data