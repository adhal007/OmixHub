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



