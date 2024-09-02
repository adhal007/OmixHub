import src.Connectors.gcp_bigquery_utils as gcp_bigquery_utils

class BigQueryEngine:
    def __init__(self, project_id) -> None:
        self.project_id = project_id
        self._gcp_utils = gcp_bigquery_utils.BigQueryUtils(project_id)
        
        pass