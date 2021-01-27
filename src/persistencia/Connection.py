from persistencia.File import File

class Connection:

    def __init__(self):
        self.File = File('token.ini',[])

    def fn_getToken(self):
        data = self.File.load_token()
        return data
        