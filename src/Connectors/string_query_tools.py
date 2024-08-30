import requests ## python -m pip install requests 
import json

class StringQuery:
    def __init__(self, method:str, output_format:str, gene_setL:list):
        self.string_api_url = "https://version-11-5.string-db.org/api"
        self.output_format = output_format
        self._methods = ["enrichment", "interaction_partners", "network" ]
        