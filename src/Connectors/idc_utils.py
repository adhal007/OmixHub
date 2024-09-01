


import pandas as pd
from google.cloud import bigquery

class TCGADataIntegrator:
    def __init__(self, rna_seq_metadata_path):
        # Load RNA-Sequencing metadata
        self.rna_seq_metadata = pd.read_csv(rna_seq_metadata_path)
        # Initialize BigQuery client
        self.client = bigquery.Client()

    ### THis should go into the queries section
    def fetch_image_data(self, modality):
        # Query to fetch image data based on modality
        query = f"""
        WITH 
        img_data AS (
          SELECT PatientID, idc_case_id, collection_id, collection_name, collection_cancerType, Modality, collection_tumorLocation  
          FROM `bigquery-public-data.idc_current.dicom_all_view`
        ),
        gdc_data AS (
          SELECT dicom_patient_id, case_gdc_id 
          FROM `bigquery-public-data.idc_current_clinical.tcga_kich_clinical`
        )

        SELECT 
          img_data.PatientID,
          img_data.idc_case_id,
          img_data.collection_id,
          img_data.collection_name,
          img_data.collection_cancerType,
          img_data.Modality,
          img_data.collection_tumorLocation,
          gdc_data.case_gdc_id
        FROM 
          img_data
        JOIN 
          gdc_data
        ON 
          img_data.PatientID = gdc_data.dicom_patient_id
        WHERE 
          img_data.Modality = '{modality}';
        """
        # Run the query
        image_data = self.client.query(query).to_dataframe()
        return image_data

    def combine_data(self, image_data):
        # Merge the image data with RNA-Sequencing metadata
        combined_data = pd.merge(image_data, self.rna_seq_metadata, left_on='case_gdc_id', right_on='Case ID', how='inner')
        return combined_data

    def get_combined_data_for_modality(self, modality):
        # Fetch image data for the specified modality
        image_data = self.fetch_image_data(modality)
        # Combine it with RNA-Sequencing data
        combined_data = self.combine_data(image_data)
        return combined_data

