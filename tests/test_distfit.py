'''
Created on 2022-05-17

@author: wf
'''
from tests.basetest import BaseTest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from pdffit.distfit import BestFitDistribution
import os
import sys


class TestProbabilityDistributionFit(BaseTest):
    '''
    test the ProbabilityDistributionFit
    '''
    
    def setUp(self, debug=False, profile=True):
        BaseTest.setUp(self, debug=debug, profile=profile)
        self.outputroot="/tmp/pdfit"
        os.makedirs(self.outputroot,exist_ok=True)
            
    def testZipf(self):
        '''
        test the Zipf distribution
        '''
        show=False
        for a in [1.2,1.4,1.6]:
            x = np.random.zipf(a=a, size=1000)
            xlog=np.log(x)
            df = pd.Series(xlog) 
            # distributionNames=["powerlaw","norm"]
            
            bfd=BestFitDistribution(df)
            bfd.analyze(f"Zipf distribution a={a:.1f}", x_label="x", y_label="zipf(x,a)",density=False,outputFilePrefix=f"/tmp/zipf{a}")
        if show:
            plt.show()
        