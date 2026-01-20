import numpy as np
import pandas as pd
import os
import joblib
import pickle
import math
import ast


class Config:
    """Configuration for B and T Cells analysis paths"""
    
    def __init__(self):
        """
        Initialize the config object.
        """
        # Base paths
        self._BCELL_BASE = '/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_1st_donor_UPMC_aggr/dictys_outs/actb1_added_v2'
        self._TCELL_BASE = '/ocean/projects/cis240075p/asachan/datasets/B_Cell/T_cell/outs/dictys/rbpj_ntc'
        self.OUTPUT_FOLDER = f'{self._BCELL_BASE}/output/figures'
        self.INPUT_FOLDER = f'{self._BCELL_BASE}/output/intermediate_tmp_files'
        self.CELL_LABELS = f'{self._BCELL_BASE}/data/clusters.csv'
        
        # State discriminative LFs
        _ENRICHMENT_BASE = f'{self.INPUT_FOLDER}/direct_effect_enrichment'
        
        self.PB = {
            'ep1': f'{_ENRICHMENT_BASE}/enrichment_ep1_pb.csv',
            'ep2': f'{_ENRICHMENT_BASE}/enrichment_ep2_pb.csv',
            'ep3': f'{_ENRICHMENT_BASE}/enrichment_ep3_pb.csv',
            'ep4': f'{_ENRICHMENT_BASE}/enrichment_ep4_pb.csv',
        }
        
        self.GC = {
            'ep1': f'{_ENRICHMENT_BASE}/enrichment_ep1_gc.csv',
            'ep2': f'{_ENRICHMENT_BASE}/enrichment_ep2_gc.csv',
            'ep3': f'{_ENRICHMENT_BASE}/enrichment_ep3_gc.csv',
            'ep4': f'{_ENRICHMENT_BASE}/enrichment_ep4_gc.csv',
        }
        
        # TF KO paths
        self.IRF4 = {
            f'ep{i}': f'{self.INPUT_FOLDER}/irf4_ko/gc_98/enrichment_episode_{i}.csv' 
            for i in range(1, 9)
        }
        
        self.BLIMP1 = {
            f'ep{i}': f'{self.INPUT_FOLDER}/prdm1_ko/gc_98/enrichment_episode_{i}.csv' 
            for i in range(1, 9)
        }
        
        self.ETS1_SIG = {
            f'ep{i}': f'{self._TCELL_BASE}/output/ets1/sig_LFs/enrichment_episode_{i}.csv' 
            for i in range(1, 7)
        }
        
        self.ETS1_ALL = {
            f'ep{i}': f'{self._TCELL_BASE}/output/ets1/all_lfs/enrichment_episode_{i}.csv' 
            for i in range(1, 7)
        }
        
        self.IKZF1_SIG = {
            f'ep{i}': f'{self._TCELL_BASE}/output/ikzf1/sig_lfs/enrichment_episode_{i}.csv' 
            for i in range(1, 7)
        }
        
        self.IKZF1_ALL = {
            f'ep{i}': f'{self._TCELL_BASE}/output/ikzf1/all_lfs/enrichment_episode_{i}.csv' 
            for i in range(1, 7)
        }