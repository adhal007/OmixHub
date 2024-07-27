import streamlit as st 
import src.Connectors.mongo_db_conn as mongo_db_conn
import pandas as pd 
#App Title
st.title('RNASeq Metadata Visualization😎')

@st.cache_resource
def init_connection():
    # connection_string = st.secrets["mongo"]["connection_string"]
    
    db = "GDCSequencingData"
    collection = "RNASeq2"
    # conn_str = "mongodb+srv://omicsmlhub:<password>@gdcquerydata.s740kfp.mongodb.net/?retryWrites=true&w=majority&appName=GDCQueryData"

    mongo_db_inst = mongo_db_conn.MongoDBConnector(connection_str=None, db=db, collection=collection)
    return mongo_db_inst

client = init_connection()
db = client._db

@st.cache_data(ttl=600)
def get_all_metadata():
    collection = db["RNASeq2"].find({})
    all_metadata = pd.DataFrame(collection)
    return all_metadata

@st.cache_data(ttl=600)
def display_unique_primary_diagnosis_and_count():
    all_meta = get_all_metadata()
    all_meta_counts = all_meta['primary_diagnosis'].value_counts().reset_index()
    all_meta_counts.columns = ['primary_diagnosis', 'count']
    return all_meta
    
#disply the dataframe
ps_data = display_unique_primary_diagnosis_and_count()
st.write(ps_data)