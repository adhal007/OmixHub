import pandas as pd
import numpy as np
import os 
from shutil import move
import src.CustomLogger.custom_logger
import src.Connectors.gdc_parser as gdc_prs

class EndptPreProcessor(gdc_prs.GDCParser):
    def __init__(self):
        super().__init__()
        pass 

    def create_data_matrix(self):
        raise NotImplementedError("Method not implemented yet")
    
    def get_patient_overlap(self):
        """
        Args: Dataframe of patient data

        Returns: Dictionary of Count of different groups of categortical variables that and their patient overlap distribution 

        pat_ovr = {"Sample type": pd.DataFrame, "Project Type": pd.DataFrame, "Sample type": pd.DataFrame}
        """
        raise NotImplementedError("Method not implemented yet")
    
    