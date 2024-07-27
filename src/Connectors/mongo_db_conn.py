from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import MongoDBAtlasVectorSearch
# from langchain_openai import ChatOpenAI

class MongoDBConnector:
    def __init__(self,  connection_str, db, collection)->None:
        self.conn_str = connection_str
        self._passwd = "WQrXObraMlsi6zvt"
        if connection_str is None:
            self.uri =  f"mongodb+srv://omicsmlhub:{self._passwd}@gdcquerydata.s740kfp.mongodb.net/?retryWrites=true&w=majority&appName=GDCQueryData"
        else:
            self.uri = self.conn_str
        self._client = MongoClient(self.uri, server_api=ServerApi('1'))
        self._db = self._client[db]

        self._atlas_search_index_name = "RAGDataIndex"
        if collection is not None:
            ## check if collection exists if not create one
            self.create_collection(collection) 
            self._collection = self._db[collection]
        else:
            self._collection = None

            
        # self._db = self._client["GDCSequencingData"]
        # self._collection = self._db["RNASeq"]

    def create_collection(self, collection_name, validator=None):
        if collection_name not in self._db.list_collection_names():
            if validator:
                self._db.create_collection(collection_name, validator=validator)
            else:
                self._db.create_collection(collection_name)
        else:
            print(f"Collection '{collection_name}' already exists.") 
            
    def insert_df_to_mongo(self, data_frame):
        collection = self._collection
        records = data_frame.to_dict(orient='records')
        collection.insert_many(records)
        return 

    def insert_records_to_mongo(self, records):
        collection = self._collection
        collection.insert_many(records)
        return 
    
    def ping(self):
        try:
            self._client.admin.command(command='ping')
            print("Pinged your deployment. You successfully connected to MongoDB!")
        except Exception as e:
            print(e)
            
    def create_collection(self, collection_name, validator=None):
        if collection_name not in self._db.list_collection_names():
            if validator:
                self._db.create_collection(collection_name, validator=validator)
            else:
                self._db.create_collection(collection_name)
        else:
            print(f"Collection '{collection_name}' already exists.") 
            

    def create_docs(self, context_list, corpus_term):
        corpus_docs = []
        for context in context_list:
            corpus_docs.append({"term": corpus_term, "context": context})
        return corpus_docs
    
    # def create_vector_search_from_texts(self, context_list, curated_term_list):
    #     """
    #     Creates a MongoDBAtlasVectorSearch object using the connection string, database, and collection names, along with the OpenAI embeddings and index configuration.

    #     :return: MongoDBAtlasVectorSearch object
    #     """
    #     vector_search = MongoDBAtlasVectorSearch.from_texts(
    #         texts=[context_list],
    #         embedding=OpenAIEmbeddings(disallowed_special=()),
    #         collection=self.collection,
    #         metadatas=[{'curated_term': curated_term_list}],
    #         index_name=self._atlas_search_index_name,
    #     )
    #     return vector_search
