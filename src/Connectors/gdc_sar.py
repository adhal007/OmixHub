import json
import requests

class GDCClient:
    def __init__(self, homepage='https://api.gdc.cancer.gov'):
        self.homepage = homepage
        self.headers = {
            'Content-Type': 'application/json'
        }

    def query(self, endpoint, params=None, method='GET', data=None):
        """
        General purpose method to query GDC API.
        """
        url = f"{self.homepage}{endpoint}"
        response = requests.request(method, url, headers=self.headers, params=params, data=json.dumps(data) if data else data)
        if response.status_code == 200:
            try:
                return response.json()
            except json.JSONDecodeError as e:
                print(f"Failed to decode JSON. Status code: {response.status_code}, Response text: {response.text}")
                raise
        else:
            print(f"Error: {response.status_code}, Response: {response.text}")
            response.raise_for_status()

    def search(self, endpoint, filters, fields, size=100, format='json'):
        """
        Search data in the GDC using filters and expansion, now including format handling.
        """
        data = {
            'filters': json.dumps(filters),
            'fields': ','.join(fields),
            'format': format,
            'size': size
        }
        return self.query(endpoint, method='POST', data=data)

    def get_project(self, project_id):
        """
        Retrieve details for a specific project.
        """
        return self.query(f'/projects/{project_id}', method='GET')

    def list_projects(self, fields=None, size=10, page=1):
        """
        List available projects with optional fields.
        """
        return self.search('/projects', fields=fields, size=size, page=page)

    def get_case(self, case_id):
        """
        Retrieve details for a specific case.
        """
        return self.query(f'/cases/{case_id}', method='GET')

    def list_cases(self, filters, fields=None, expand=None, size=10, page=1):
        """
        Retrieve cases based on filters with optional expansion.
        """
        return self.search('/cases', filters=filters, fields=fields, expand=expand, size=size, page=page)

    def get_clinical_info(self, case_id):
        """
        Retrieve clinical information for a specific case.
        """
        return self.query(f'/cases/{case_id}?expand=diagnoses,treatments', method='GET')


