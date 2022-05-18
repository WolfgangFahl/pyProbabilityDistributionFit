'''
Created on 2022-05-17

see 
  https://stackoverflow.com/questions/6620471/fitting-empirical-distribution-to-theoretical-ones-with-scipy-python/37616966#37616966

@author: https://stackoverflow.com/users/832621/saullo-g-p-castro
see https://stackoverflow.com/a/37616966/1497139

@author: https://stackoverflow.com/users/2087463/tmthydvnprt
see  https://stackoverflow.com/a/37616966/1497139

@author: https://stackoverflow.com/users/1497139/wolfgang-fahl
see 

'''
import traceback
import sys
import warnings

import numpy as np
import pandas as pd
import scipy.stats as st

from typing import Callable
from scipy.stats._continuous_distns import _distn_names

import statsmodels.api as sm

import matplotlib
import matplotlib.pyplot as plt

class BestFitDistribution():
    '''
    Find the best Probability Distribution Function for the given data
    '''
    
    def __init__(self,data,backend:str="WebAgg",distributionNames:list=None,debug:bool=False):
        '''
        constructor
        
        Args:
            data(dataFrame): the data to analyze
            distributionNames(list): list of distributionNames to try
            debug(bool): if True show debugging information
        '''
        self.debug=debug
        self.backend=backend
        self.matplotLibParams()
        if distributionNames is None:
            self.distributionNames=[d for d in _distn_names if not d in ['levy_stable', 'studentized_range']]
        else:
            self.distributionNames=distributionNames
        self.data=data
        
    def matplotLibParams(self):
        '''
        set matplotlib parameters
        '''
        matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
        matplotlib.style.use('ggplot')
        matplotlib.use(self.backend)

    # Create models from data
    def best_fit_distribution(self,bins:int=200, ax=None,density:bool=True):
        """
        Model data by finding best fit distribution to data
        """
        # Get histogram of original data
        y, x = np.histogram(self.data, bins=bins, density=density)
        x = (x + np.roll(x, -1))[:-1] / 2.0
    
        # Best holders
        best_distributions = []
        distributionCount=len(self.distributionNames)
        # Estimate distribution parameters from data
        for ii, distributionName in enumerate(self.distributionNames):
    
            print(f"{ii+1:>3} / {distributionCount:<3}: {distributionName}")
    
            distribution = getattr(st, distributionName)
    
            # Try to fit the distribution
            try:
                # Ignore warnings from data that can't be fit
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    
                    # fit dist to data
                    params = distribution.fit(self.data)
    
                    # Separate parts of parameters
                    arg = params[:-2]
                    loc = params[-2]
                    scale = params[-1]
                    
                    # Calculate fitted PDF and error with fit in distribution
                    pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                    sse = np.sum(np.power(y - pdf, 2.0))
                    
                    # if axis pass in add to plot
                    try:
                        if ax:
                            pd.Series(pdf, x).plot(ax=ax)
                    except Exception:
                        pass
    
                    # identify if this distribution is better
                    best_distributions.append((distribution, params, sse))
            
            except Exception as ex:
                if self.debug:
                    trace=traceback.format_exc()
                    msg=f"fit for {distributionName} failed:{ex}\n{trace}"
                    print(msg,file=sys.stderr)
                pass
        
        return sorted(best_distributions, key=lambda x:x[2])

    def make_pdf(self,dist, params:list, size=10000):
        """
        Generate distributions's Probability Distribution Function 
        
        Args:
            dist: Distribution
            params(list): parameter
            size(int): size
            
        Returns:
            dataframe: Power Distribution Function 
        
        """
    
        # Separate parts of parameters
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]
    
        # Get sane start and end points of distribution
        start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
        end   = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)
    
        # Build PDF and turn into pandas Series
        x = np.linspace(start, end, size)
        y = dist.pdf(x, loc=loc, scale=scale, *arg)
        pdf = pd.Series(y, x)
    
        return pdf
    
    def analyze(self,title,x_label,y_label,facecolor='b',alpha=0.5,callback:Callable=None,outputFilePrefix=None,imageFormat:str='png',allBins:int=50,distBins:int=200,density:bool=True):
        """
        
        analyze the Probabilty Distribution Function
        
        Args:
            data: Panda Dataframe or numpy array
            title(str): the title to use
            x_label(str): the label for the x-axis
            y_label(str): the label for the y-axis
            
            facecolor(str): the color to use
            alpha(float): the opacity to use
            
            callback(Callable): a function to be called for the plots
            
            outputFilePrefix(str): the prefix of the outputFile
            imageFormat(str): imageFormat e.g. png,svg
            
            allBins(int): the number of bins for all
            distBins(int): the number of bins for the distribution
            density(bool): if True show relative density
        """
        self.allBins=allBins
        self.distBins=distBins
        self.density=density
        self.title=title
        self.x_label=x_label
        self.y_label=y_label
        self.facecolor=facecolor
        self.alpha=alpha
        self.callback=callback
        self.imageFormat=imageFormat
        self.outputFilePrefix=outputFilePrefix
        self.best_dist=None
        self.analyzeAll()
        if self.callback:
            self.callback(self.figAll,isAll=True)
        if outputFilePrefix is not None:
            self.saveFig(f"{outputFilePrefix}All.{imageFormat}", imageFormat)
            plt.close(self.figAll)
        if self.best_dist:
            self.analyzeBest()
            if self.callback:
                self.callback(self.figBest,isAll=False)
            if outputFilePrefix is not None:
                self.saveFig(f"{outputFilePrefix}Best.{imageFormat}", imageFormat)
                plt.close(self.figBest)
            
    def analyzeAll(self):
        '''
        analyze the given data
        
        '''
        # Plot for comparison
        figTitle=f"{self.title}\n All Fitted Distributions"
        self.figAll=plt.figure(figTitle,figsize=(12,8))
        ax = self.data.plot(kind='hist', bins=self.allBins, density=self.density, alpha=self.alpha, facecolor=self.facecolor)
        
        # Save plot limits
        dataYLim = ax.get_ylim()
        # Update plots
        ax.set_ylim(dataYLim)
        ax.set_title(figTitle)
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)
        
        # Find best fit distribution
        best_distributions = self.best_fit_distribution(bins=self.distBins, ax=ax,density=self.density)
        if len(best_distributions)>0:
            self.best_dist = best_distributions[0]
            # Make PDF with best params 
            self.pdf = self.make_pdf(self.best_dist[0], self.best_dist[1])
        
    def analyzeBest(self):
        '''
        analyze the Best Property Distribution function
        '''
        # Display
        figLabel="PDF"
        self.figBest=plt.figure(figLabel,figsize=(12,8))
        ax = self.pdf.plot(lw=2, label=figLabel, legend=True)
        self.data.plot(kind='hist', bins=self.allBins, density=self.density, alpha=self.alpha, label='Data', legend=True, ax=ax,facecolor=self.facecolor)
        
        param_names = (self.best_dist[0].shapes + ', loc, scale').split(', ') if self.best_dist[0].shapes else ['loc', 'scale']
        param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, self.best_dist[1])])
        dist_str = '{}({})'.format(self.best_dist[0].name, param_str)
        
        ax.set_title(f'{self.title} with best fit distribution \n' + dist_str)
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)

    def saveFig(self,outputFile:str=None,imageFormat='png'):
        '''
        save the current Figure to the given outputFile
        
        Args:
            outputFile(str): the outputFile to save to
            imageFormat(str): the imageFormat to use e.g. png/svg
        '''
        plt.savefig(outputFile, format=imageFormat) # dpi 
     

if __name__ == '__main__':
    # Load data from statsmodels datasets
    data = pd.Series(sm.datasets.elnino.load_pandas().data.set_index('YEAR').values.ravel())
    #color=list(matplotlib.rcParams['axes.prop_cycle'])[1]['color']
   
    bfd=BestFitDistribution(data)
    for density in [True,False]:
        suffix="density" if density else ""
        bfd.analyze(title=u'El Niño sea temp.',x_label=u'Temp (°C)',y_label='Frequency',outputFilePrefix=f"/tmp/ElNinoPDF{suffix}",density=density)
    # uncomment for interactive display
    # plt.show()
