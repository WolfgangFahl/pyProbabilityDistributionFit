"""
Created on 2022-05-17

@author: wf
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import zeta

# from collections import Counter
from pdffit.distfit import BestFitDistribution
from tests.basetest import BaseTest

# import sys


class TestProbabilityDistributionFit(BaseTest):
    """
    test the ProbabilityDistributionFit
    """

    def setUp(self, debug=False, profile=True):
        BaseTest.setUp(self, debug=debug, profile=profile)
        self.outputroot = "/tmp/pdfit"
        os.makedirs(self.outputroot, exist_ok=True)
        np.random.seed(0)

    def testZipf(self):
        """
        test the Zipf distribution
        """

        def callback(fig, isAll: bool):
            """
            modify the histogramm
            """
            plt.semilogy()
            pass

        show = False
        datapoints = 1000
        for a in [4.0]:
            x = np.random.default_rng().zipf(a, size=datapoints)
            df = pd.DataFrame({"zipf": x})
            # distributionNames=["powerlaw","norm"]

            bfd = BestFitDistribution(df)
            bfd.analyze(
                f"Zipf distribution a={a:.1f}",
                x_label="x",
                y_label="zipf(x,a)",
                density=True,
                callback=callback,
                outputFilePrefix=f"{self.outputroot}/zipf{a}",
            )
        if show:
            plt.show()

    def testZipfDisplay(self):
        """
        test the Zipf Distribution display
        """
        show = not self.inPublicCI()
        if show:
            matplotlib.use("WebAgg")
        # https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.zipf.html#numpy.random.Generator.zipf
        a = 4.0
        n = 1000
        s = np.random.default_rng().zipf(a, size=n)
        count = np.bincount(s)
        k = np.arange(1, s.max() + 1)
        plt.bar(k, count[1:], alpha=0.5, label="sample count")
        plt.plot(k, n * (k**-a) / zeta(a), "k.-", alpha=0.5, label="expected count")
        plt.semilogy()
        plt.grid(alpha=0.4)
        plt.legend()
        plt.title(f"Zipf sample, a={a}, size={n}")
        if show:
            plt.show()

    def testNormal(self):
        """
        test the normal distribution
        """
        # use euler constant as seed

        # statistically relevant number of datapoints
        datapoints = 1000
        a = np.random.normal(40, 10, datapoints)
        df = pd.DataFrame({"nums": a})
        outputFilePrefix = f"{self.outputroot}/normalDist"
        bfd = BestFitDistribution(df, debug=True)
        bfd.analyze(
            title="normal distribution",
            x_label="x",
            y_label="random",
            outputFilePrefix=outputFilePrefix,
        )
