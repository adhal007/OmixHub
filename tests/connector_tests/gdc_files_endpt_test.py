import unittest
from src.Connectors.gdc_files_endpt import GDCFilesEndpt

class TestGDCFilesEndpt(unittest.TestCase):
    def setUp(self):
        self.gdc_files_endpt = GDCFilesEndpt()

    def test_fetch_rna_seq_star_counts_data(self):
        # Use some test inputs
        new_fields = ['file_id', 'analysis.metadata.read_groups.platform']
        ps_list = ['kidney', 'bronchus and lung']
        race_list = ['white']
        gender_list = ['Male', 'Female']

        # Call the method with the test inputs
        json_data, filters = self.gdc_files_endpt.fetch_rna_seq_star_counts_data(new_fields, ps_list, race_list, gender_list)

        # Use assert methods to verify the output
        self.assertIsInstance(json_data, dict)
        self.assertIsInstance(filters, dict)

        # Add more asserts as necessary to check the contents of json_data and filters

    # Repeat for other methods in GDCFilesEndpt

if __name__ == '__main__':
    unittest.main()