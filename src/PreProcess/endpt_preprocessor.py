import pandas as pd
import numpy as np
import os 
from shutil import move
import src.CustomLogger.custom_logger
import src.Connectors.gdc_parser as gdc_prs

class EndptPreProcessor:
    def __init__(self,data:pd.DataFrame=None):
        self.data = data
        pass 

    def create_data_matrix(self):
        raise NotImplementedError("Method not implemented yet")
    
    def get_patient_overlap(self,data):
        """
        Args: Dataframe of patient data

        Returns: Dictionary of Count of different groups of categortical variables that and their patient overlap distribution 

        pat_ovr = {"Sample type": pd.DataFrame, "Project Type": pd.DataFrame, "Sample type": pd.DataFrame}
        """
        categorical_variables = ['Sample Type','Project ID']
        pat_over = {}
        if data is None:
            data=self.data
        for category in categorical_variables:
            if category in data.columns:
                category_counts = data.groupby(category)['Case ID'].nunique().reset_index()
                category_counts.columns = [category, 'Unique_Patient_Count']
                pat_over[category] = category_counts
        
        return pat_over
       
    