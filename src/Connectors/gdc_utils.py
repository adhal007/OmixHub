import json
import requests
import src.Connectors.gdc_sar as gdc_sar

# Example of usage
class GDCUtils(gdc_sar.GDCClient):
    def __init__(self, homepage='https://api.gdc.cancer.gov'):
        super().__init__(homepage)

    def fetch_case_details(self, case_id):
        """
        Fetch detailed information for a specific case.
        """
        return self.query(f"/cases/{case_id}", method='GET')

    def list_projects_by_disease(self, disease_type):
        """
        List projects filtered by disease type with specified fields.
        """
        filters = {
            "op": "=",
            "content": {
                "field": "projects.disease_type",
                "value": [disease_type]
            }
        }
        # Define which fields about the projects you are interested in retrieving.
        fields = [
            "project_id", "project_name", "primary_site", "program.name"
        ]
        return self.search('/projects', filters=filters, fields=fields)

    def search_files_by_criteria(self, primary_sites, experimental_strategies, data_formats):
        """
        Search files based on primary site, experimental strategy, and data format.
        """
        filters = {
            "op": "and",
            "content": [
                {
                    "op": "in",
                    "content": {
                        "field": "cases.project.primary_site",
                        "value": primary_sites
                    }
                },
                {
                    "op": "in",
                    "content": {
                        "field": "files.experimental_strategy",
                        "value": experimental_strategies
                    }
                },
                {
                    "op": "in",
                    "content": {
                        "field": "files.data_format",
                        "value": data_formats
                    }
                }
            ]
        }
        fields = [
            "file_id", "file_name", "cases.submitter_id", "cases.case_id",
            "data_category", "data_type", "cases.samples.tumor_descriptor",
            "cases.samples.tissue_type", "cases.samples.sample_type",
            "cases.samples.submitter_id", "cases.samples.sample_id",
            "cases.samples.portions.analytes.aliquots.aliquot_id",
            "cases.samples.portions.analytes.aliquots.submitter_id"
        ]
        return self.search('/files', filters=filters, fields=fields, format=data_formats, size=100)
    
    # def get_list_of_all_diseases(self):
        
    #     filters = {  
    #         "size":"20000",
    #         "pretty":"TRUE",
    #         "fields":"submitter_id,disease_type",
    #         "format":"TSV",
    #         "filters":{  
    #             "op":"=",
    #             "content":{  
    #                 "field":"disease_type",
    #                 "value":"*"
    #             }
    #         }
    #         }

    # def download_file(self, file_uuid):
    #     """
    #     Download a file by its UUID.
    #     """
    #     file_info = self.query(f"/files/{file_uuid}", method='GET')
    #     if 'url' in file_info:
    #         response = requests.get(file_info['url'])
    #         if response.status_code == 200:
    #             with open(file_uuid, 'wb') as f:
    #                 f.write(response.content)
    #             return f"{file_uuid} downloaded successfully."
    #         else:
    #             return "Failed to download the file."
    #     else:
    #         return "File URL not found."

    # def fetch_gene_expression_files(self, tcga_barcodes):
    #     """
    #     Retrieve Gene Expression Quantification files and associated metadata for specified TCGA cases.
    #     """
    #     filters = {
    #         "op": "and",
    #         "content": [
    #             {
    #                 "op": "in",
    #                 "content": {
    #                     "field": "cases.submitter_id",
    #                     "value": tcga_barcodes
    #                 }
    #             },
    #             {
    #                 "op": "=",
    #                 "content": {
    #                     "field": "files.data_type",
    #                     "value": "Gene Expression Quantification"
    #                 }
    #             }
    #         ]
    #     }
    #     fields = [
    #         "file_id", "file_name", "cases.submitter_id", "cases.case_id", "data_category",
    #         "data_type", "cases.samples.tumor_descriptor", "cases.samples.tissue_type",
    #         "cases.samples.sample_type", "cases.samples.submitter_id", "cases.samples.sample_id",
    #         "analysis.workflow_type", "cases.project.project_id",
    #         "cases.samples.portions.analytes.aliquots.aliquot_id",
    #         "cases.samples.portions.analytes.aliquots.submitter_id"
    #     ]
    #     return self.search('/files', filters=filters, fields=fields, format='tsv', size=1000)
    
    # def fetch_metadata_for_files(self, file_uuids):
    #     """
    #     Fetch metadata in TSV format for a set of files using their UUIDs.
    #     """
    #     filters = {
    #         "op": "in",
    #         "content": {
    #             "field": "files.file_id",
    #             "value": file_uuids
    #         }
    #     }
    #     fields = [
    #         "file_id", "file_name", "cases.submitter_id", "cases.case_id",
    #         "data_category", "data_type", "cases.samples.tumor_descriptor",
    #         "cases.samples.tissue_type", "cases.samples.sample_type",
    #         "cases.samples.submitter_id", "cases.samples.sample_id",
    #         "cases.samples.portions.analytes.aliquots.aliquot_id",
    #         "cases.samples.portions.analytes.aliquots.submitter_id"
    #     ]
    #     return self.search('/files', filters=filters, fields=fields, format='tsv', size=100)
    
    # def fetch_metadata_by_primary_sites(self, primary_sites):
    #     """
    #     Fetch metadata in TSV format for cases based on their primary sites.
    #     """
    #     filters = {
    #         "op": "in",
    #         "content": {
    #             "field": "cases.primary_site",
    #             "value": primary_sites
    #         }
    #     }
    #     fields = [
    #         "file_id", "file_name", "cases.submitter_id", "cases.case_id",
    #         "data_category", "data_type", "cases.samples.tumor_descriptor",
    #         "cases.samples.tissue_type", "cases.samples.sample_type",
    #         "cases.samples.submitter_id", "cases.samples.sample_id",
    #         "cases.samples.portions.analytes.aliquots.aliquot_id",
    #         "cases.samples.portions.analytes.aliquots.submitter_id"
    #     ]
    #     return self.search('/files', filters=filters, fields=fields, format='tsv', size=100)

    # def search_gene_expression_by_primary_site(self, primary_sites, tcga_barcodes=None):
    #     """
    #     Search for Gene Expression Quantification files for specific primary sites and optionally by TCGA barcodes.
    #     """
    #     filters = {
    #         "op": "and",
    #         "content": [
    #             {
    #                 "op": "in",
    #                 "content": {
    #                     "field": "files.data_type",
    #                     "value": ["Gene Expression Quantification"]
    #                 }
    #             },
    #             {
    #                 "op": "in",
    #                 "content": {
    #                     "field": "cases.primary_site",
    #                     "value": primary_sites
    #                 }
    #             }
    #         ]
    #     }
    #     # Add TCGA barcode filter if provided
    #     if tcga_barcodes:
    #         filters["content"].append({
    #             "op": "in",
    #             "content": {
    #                 "field": "cases.submitter_id",
    #                 "value": tcga_barcodes
    #             }
    #         })

    #     fields = [
    #         "file_id", "file_name", "cases.submitter_id", "cases.case_id",
    #         "data_category", "data_type", "cases.samples.tumor_descriptor",
    #         "cases.samples.tissue_type", "cases.samples.sample_type",
    #         "cases.samples.submitter_id", "cases.samples.sample_id",
    #         "cases.samples.portions.analytes.aliquots.aliquot_id",
    #         "cases.samples.portions.analytes.aliquots.submitter_id"
    #     ]
    #     return self.search('/files', filters=filters, fields=fields, format='tsv', size=100)



# class search_and_retrieval:
#     def __init__(self, **params) -> None:

#         self.list_of_endpts = ['files', 'cases', 'history', 'projects', 'annotation', '_mapping']
#         self.gdc_home = 'https://api.gdc.cancer.gov/'
#         self.request_params = ['filters', 'format', 'pretty', 'fields', 'expand', 'size', 'from', 'sort', 'facets']

#         self.default_params = {
#             'endpt_type': 'cases'
            
#             ''
#         }

# class gdc_conn:
#     def __init__(self, path_to_save, fields, filters) -> None:
#         self.local_path = path_to_save
#         self.fields = fields
#         self.filters = filters 

#     def check_connection(self):
#         status_endpt = "https://api.gdc.cancer.gov/status"
#         response = requests.get(status_endpt)
#         print(response.content)
#         return
     
#     def download_bam_files(self, file_ids):
#         token_file = "$TOKEN_FILE_PATH"

#         file_ids = [
#             "11443f3c-9b8b-4e47-b5b7-529468fec098",
#             "1f103620-bb34-46f1-b565-94f0027e396d",
#             "ca549554-a244-4209-9086-92add7bb7109"
#             ]

#         for file_id in file_ids:

#             data_endpt = "https://api.gdc.cancer.gov/slicing/view/{}".format(file_id)

#             with open(token_file, "r") as token:
#                 token_string = str(token.read().strip())

#             params = {
#                 "regions": ["chr1:1-20000", "chr10:129000-160000"]
#                 }

#             response = requests.post(data_endpt,
#                                     data = json.dumps(params),
#                                     headers = {
#                                         "Content-Type": "application/json",
#                                         "X-Auth-Token": token_string
#                                         })

#             file_name = "{}_region_slices.bam".format(file_id)

#             with open(file_name, "wb") as output_file:
#                 output_file.write(response.content)

#     def download_study_metadata(self, file_name):
#         fields = [
#             "file_name",
#             "cases.submitter_id",
#             "cases.samples.sample_type",
#             "cases.disease_type",
#             "cases.project.project_id"
#             ]

#         fields = ",".join(fields)

#         files_endpt = "https://api.gdc.cancer.gov/files"

#         # This set of filters is nested under an 'and' operator.
#         filters = {
#             "op": "and",
#             "content":[
#                 {
#                 "op": "in",
#                 "content":{
#                     "field": "cases.project.primary_site",
#                     "value": ["Lung"]
#                     }
#                 },
#                 {
#                 "op": "in",
#                 "content":{
#                     "field": "files.experimental_strategy",
#                     "value": ["RNA-Seq"]
#                     }
#                 },
#                 {
#                 "op": "in",
#                 "content":{
#                     "field": "files.data_format",
#                     "value": ["BAM"]
#                     }
#                 }
#             ]
#         }

#         # A POST is used, so the filter parameters can be passed directly as a Dict object.
#         params = {
#             "filters": filters,
#             "fields": fields,
#             "format": "TSV",
#             "size": "2000"
#             }

#         # The parameters are passed to 'json' rather than 'params' in this case
#         response = requests.post(files_endpt, headers = {"Content-Type": "application/json"}, json = params)
#         with open(file_name, "wb") as output_file:
#                 output_file.write(response.content.decode("utf-8"))
#         # print(response.content.decode("utf-8"))
        
#     def download_single_file(self, uuid_file, filename):
#         response, filename = Data.download(uuid=uuid_file, path=self.local_path, name=filename)
#         return filename
