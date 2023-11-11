import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

class PyDeSeqWrapper:
    def __init__(self, count_matrix:pd.DataFrame, **kwargs)->None:
        self.count_matrix = count_matrix
        self.kwargs = kwargs

        self._metadata = None
        self._design_factors = None
        self._dds = None 

    @property
    def dds(self):
        if self._dds is None:
            self._dds = self._get_dds()
        return self._dds
    
    @property
    def design_factors(self):
        if self._design_factors is None:
            return self.kwargs.get('design_factors')
    
    @property
    def metadata(self):
        if self._metadata is None:
            metadata = self.kwargs.get('metadata') 
        return metadata
    
    def _get_dds(self):
        dds = DeseqDataSet(counts=self.count_matrix, metadata=self.metadata, design_factors=self.design_factors)
        dds.deseq2()
        return dds 
    
    def _get_deseq_stats(self):
        ## calculate results
        dss = DeseqStats(dds=self.dds, n_cpus=4, contrast = (self.design_factors,self.kwargs.get('group1'),self.kwargs.get('group2')))
        return dss
    
    def run_deseq(self):
        dss = self._get_deseq_stats()
        stat_res = dss.summary()
        res = stat_res.results_df
        return res, dss
    
    