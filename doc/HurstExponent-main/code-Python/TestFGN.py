#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 13:52:42 2023

@author: zqfeng
"""

import warnings
import numpy as np
from fgn import fgn
import pandas as pd
from HurstIndexSolver import HurstIndexSolver

warnings.filterwarnings("ignore")


def feval(func: str, para: str, Class=""):
    if len(Class) != 0:
        Class += "."
    func = Class + func + "({:s})"
    return eval(func.format(para))


def GenFGN(N, m, Hlist):
    for H in Hlist:
        fH = {}
        for i in range(1, m + 1):
            fH["f" + str(i)] = fgn(N, H)
        df = pd.DataFrame(fH)
        df.to_csv("./data/FGN_{:.2f}_{:d}.csv".format(H, m), index=False)
    return None


N, m = 30000, 30
Hlist = np.linspace(0.3, 0.8, 11)
# GenFGN(N, m, Hlist)

Methods = {
    'AM'  : "EstHurstAbsoluteMoments",
    'AV'  : "EstHurstAggregateVariance",
    'GHE' : "EstHurstGHE",
    'HM'  : "EstHurstHiguchi",
    'DFA' : "EstHurstDFAnalysis",
    'HM'  : "EstHurstHiguchi",
    'VRR' : "EstHurstRegrResid",
    'RS'  : "EstHurstRSAnalysis",
    'RS2' : "RS4Hurst",
    'TTA' : "EstHurstTTA",
    'PM'  : "EstHurstPeriodogram",
    'AWC' : "EstHurstAWC",
    'VVL' : "EstHurstVVL",
    'LW'  : "EstHurstLocalWhittle",
    'LSSD': "EstHurstLSSD",
    'LSV' : "EstHurstLSV"
    }

HSolver = HurstIndexSolver()

method = 'AM'
addParas = ""
addParas = ', minimal=50'
# addParas = ", max_scale=100"
# addParas = ', minimal=50, IsRandom=True'
# addParas = ', wavelet="haar", wavemode="periodization"'

result = []
print(Methods[method])
for H in Hlist:
    h = []
    file = "./data/FGN_{:.2f}_30.csv".format(H)
    fH = pd.read_csv(file).values
    for i in range(m):
        ts = fH[:, i]
        htmp = feval(Methods[method], "ts" + addParas, Class="HSolver")
        h.append(htmp)
    result.append([h, H])
    print("{:.4f}".format(np.mean(h)))
    # print("{:.4f} {:.2f}".format(np.mean(h), H))
