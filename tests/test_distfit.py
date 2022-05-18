'''
Created on 2022-05-17

@author: wf
'''
from tests.basetest import BaseTest
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
#from collections import Counter
from pdffit.distfit import BestFitDistribution
import os
#import sys


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
            
    def testNormal(self):
        '''
        test the normal distribution
        '''
        # use euler constant as seed
        np.random.seed(0)
        # statistically relevant number of datapoints
        datapoints=1000
        a = np.random.normal(40, 10, datapoints)
        df= pd.DataFrame({'nums':a})
        outputFilePrefix="/tmp/normalDist"
        bfd=BestFitDistribution(df,debug=True)
        bfd.analyze(title="normal distribution",x_label="x",y_label="random",outputFilePrefix=outputFilePrefix)
        