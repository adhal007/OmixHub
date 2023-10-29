import pandas as pd
import numpy as np
import os 
from shutil import move
import CustomLogger.custom_logger
logger = CustomLogger.custom_logger.CustomLogger()

class PreProcessor:
    def __init__(self):
        self.logger =  logger.custlogger(loglevel='DEBUG')
        self.logger.debug("Initialized PreProcessor Class for TCGA data")

    def remove_empty_folders(self, directory_path):
        for root, dirs, files in os.walk(directory_path, topdown=False):
            for folder in dirs:
                folder_path = os.path.join(root, folder)
                if not os.listdir(folder_path):
                    os.rmdir(folder_path)

    def get_sample_ids_folder_names_file_names(self, main_dir):
        sample_ids = []
        folder_paths = []
        for root, dirs, files in os.walk(main_dir, topdown=False):
            for folder in dirs:
                folder_path = os.path.join(root, folder)
                if os.listdir(folder_path):
                    sample_ids.append(folder)
                    folder_paths.append(folder_path)
        return sample_ids, folder_paths 
    
    def get_list_of_tsv_files(self, main_dir):
        file_names = []
        sample_ids, folder_paths = self.get_TGCA_folders_as_sample_ids(main_dir)
        for folder in folder_paths:
            file_name = os.listdir(folder)[0]

        