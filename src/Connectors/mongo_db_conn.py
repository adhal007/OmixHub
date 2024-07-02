from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi


class MongoDBConnector:
    def __init__(self, uri: str=None):
        self._passwd = "WQrXObraMlsi6zvt"
        if uri is None:
            self.uri =  f"mongodb+srv://omicsmlhub:{self._passwd}@gdcquerydata.s740kfp.mongodb.net/?retryWrites=true&w=majority&appName=GDCQueryData"
        else:
            self.uri = uri
            
        self._client = MongoClient(host=self.uri, server_api=ServerApi('1'))
        self._db = self._client["GDCSequencingData"]
        self._collection = self._db["RNASeq"]
    
    def insert_multiple(self):
        document_list = [
        { "<field name>" : "<value>" },
        { "<field name>" : "<value>" }
        ]
        result = self._collection.insert_many(document_list)
        print(result.acknowledged)
        
    def ping(self):
        try:
            self._client.admin.command(command='ping')
            print("Pinged your deployment. You successfully connected to MongoDB!")
        except Exception as e:
            print(e)
