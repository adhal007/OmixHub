import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
import OmicsUtils.CustomLogger.custom_logger as clog 
import os 


logger = clog.CustomLogger()

class PyDeSeqWrapper:
    def __init__(self, count_matrix:pd.DataFrame, **kwargs)->None:
        self.count_matrix = count_matrix
        self.kwargs = kwargs
        self.logger = logger.custlogger(loglevel='INFO')
        
        self._metadata = None
        self._design_factors = None
        self._dds = None 
        self._groups = None
        self._output_path = None 

    @property
    def output_path(self):
        if self._output_path is None:
            if  self.kwargs.get('output_path') is None:
                self._output_path = os.getcwd()
            else:
                self._output_path = self.kwargs.get('output_path')
        return self._output_path
    
    @property
    def groups(self):
        if self._groups is None:
            self._groups = self.kwargs.get('groups')
        return self._groups        

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
        dds = DeseqDataSet(counts=self.count_matrix, metadata=self.metadata, design_factors=self.design_factors, refit_cooks=True)
        dds.deseq2()
        return dds 
    
    def _get_deseq_stats(self, design_factor:str=None, group1:str=None, group2:str=None):
        ## calculate results
        dss = DeseqStats(dds=self.dds, n_cpus=4, contrast = (design_factor,group1,group2))
        return dss
    
    def run_deseq(self, design_factor, group1, group2):
        logger = self.logger.getChild('run_deseq')
        logger.info(msg=f'Running DESeq2 for groups: {self.groups}')
        if len(self.design_factors) == 1:
            logger.info(msg=f'Running DESeq2 Single factor analysis with design factor: {self.design_factors[0]}')
        elif len(self.design_factors) > 1:
            logger.info(msg=f'Running DESeq2  factor analysis with design factor: {self.design_factors[0]} and {self.design_factors[1]}')
        elif len(self.design_factors) == 0:
            raise ValueError('No design factor provided')

        logger.info(f'Statistical analysis of {group1} vs {group2} in {self.groups}')
        stat_res_group1_vs_group2  = self._get_deseq_stats(design_factor=design_factor, group1=group1, group2=group2)

        # stat_res = stat_res_group1_vs_group2.summary()
        # stat_res.results_df.to_csv(os.path.join(self.output_path, "results.csv"))
        return stat_res_group1_vs_group2 

    
    