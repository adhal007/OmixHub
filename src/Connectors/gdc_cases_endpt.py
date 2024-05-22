
import src.Connectors.gdc_utils as gdc_utils
import src.Connectors.gdc_filters as gdc_flt
import src.Connectors.gdc_fields as gdc_fld
import json

"""
Copyright (c) 2024 OmixHub.  All rights are reserved.
GDC Projects Endpoint Class and high-level API functions

@author: Abhilash Dhal
@date:  2024_23_27
"""

class GDCCasesEndpt(gdc_utils.GDCUtilsBase):
    def __init__(self, homepage='https://api.gdc.cancer.gov', endpt='cases'):
        super().__init__(homepage, endpt='cases')
        # if self.check_valid_endpt():
            self.gdc_flt = gdc_flt.GDCFilters(self.endpt)
            self.gdc_fld = gdc_fld.GDCQueryFields(self.endpt)

######### APPLICATION ORIENTED python functions for cases endpoint ################################################
################################################################################################
