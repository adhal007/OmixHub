
from src.Connectors.gdc_cases_endpt import GDCCasesEndpt
import src.Connectors.gdc_endpt_base as gdc_utils 
import src.Connectors.gdc_field_validator as gdc_vd
import src.Connectors.gdc_projects_endpt as gdc_proj
import src.deprecated.gdc_files_endpt as gdc_files
import src.Connectors.gdc_parser as gdc_prs
import json
import requests 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 

class CohortEDA(gdc_prs.GDCJson2DfParser):
    def __init__(self, gdc_files_sub: gdc_files.GDCFilesEndpt, gdc_cases_sub: GDCCasesEndpt, gdc_projs_sub: gdc_proj.GDCProjectsEndpt) -> None:
        super().__init__(gdc_files_sub, gdc_cases_sub, gdc_projs_sub)


    def primary_diagnosis_hist(self, rna_star_count_data, title):
        df =  pd.DataFrame(rna_star_count_data['cases.diagnoses.primary_diagnosis'].value_counts()).reset_index() 
            # Plotting using seaborn
        plt.figure(figsize=(10, 6))
        sns.barplot(x='count', y='cases.diagnoses.primary_diagnosis', data=df, palette='viridis')
        plt.title(f'Frequency of Primary Diagnoses in {title} Cancer Cases')
        plt.xlabel('Frequency')
        plt.ylabel('Diagnosis')
        plt.show()

    def therapy_hist(self, rna_star_count_data, title):
        """
        Function to show distribution of treatment or therapy received
        """
        df =  pd.DataFrame(rna_star_count_data['treatment_or_therapy'].value_counts()).reset_index() 
            # Plotting using seaborn
        plt.figure(figsize=(10, 6))
        sns.barplot(x='count', y='treatment_or_therapy', data=df, palette='viridis')
        plt.title(f'Frequency of treatment status in {title} Cancer Cases')
        plt.xlabel('Frequency')
        plt.ylabel('Diagnosis')
        plt.show()
   
    def days_to_last_follow_up(self, rna_star_count_data, title):
        df =  pd.DataFrame(rna_star_count_data['cases.diagnoses.days_to_last_follow_up'].value_counts()).reset_index() 
            # Plotting using seaborn
        plt.figure(figsize=(10, 6))
        sns.barplot(y='count', x='cases.diagnoses.days_to_last_follow_up', data=df, palette='viridis')
        plt.title(f'Frequency of followed up clients in {title} Cancer Cases')
        plt.xlabel('Number of patients')
        plt.ylabel('Days to last follow up')
        plt.show() 
