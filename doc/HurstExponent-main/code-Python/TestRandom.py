#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 16:39:01 2023

@author: zqfeng
"""

import numpy as np
import pandas as pd
from HurstIndexSolver import HurstIndexSolver


def feval(func: str, para: str, Class=""):
    if len(Class) != 0:
        Class += "."
    func = Class + func + "({:s})"
    return eval(func.format(para))


def GenRandom(N, m):
    random = ["normal", "chisquare", "geometric",
              "poisson", "exponential", "uniform"]
    r = [{} for i in range(6)]
    for i in range(1, m + 1):
        r[0]["N{:d}".format(i)] = np.random.randn(1, N)[0]  # N(0,1)
        r[1]["X{:d}".format(i)] = np.random.chisquare(1, N)  # X^2(1)
        r[2]["GE{:d}".format(i)] = np.random.geometric(0.25, N)  # GE(0. 25)
        r[3]["P{:d}".format(i)] = np.random.poisson(5, N)  # P(5)
        r[4]["Exp{:d}".format(i)] = np.random.exponential(1, N)  # Exp(1)
        r[5]["U{:d}".format(i)] = np.random.uniform(0, 1, N)  # U(0,1)
    for i in range(6):
        df = pd.DataFrame(r[i])
        df.to_csv("./data/Random_{:s}_{:d}.csv".format(random[i], m),
                  index=False)
    return None


HSolver = HurstIndexSolver()
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

N, m = 10000, 30
# GenRandom(N, m)
h = np.zeros([6, m])
random = ["normal", "chisquare", "geometric",
          "poisson", "exponential", "uniform"]

Class = "HSolver"
method = 'HM'
addParas = ""
addParas = ', minimal=20'
# addParas = ', minimal=50, IsRandom=True'
# addParas = ", max_scale=100"

r = [0 for i in range(6)]
for i in range(6):
    r[i] = pd.read_csv("./data/Random_{:s}_30.csv".format(random[i])).values
for i in range(m):
    # N(0, 1)
    h[0, i] = feval(Methods[method], "r[0][:, i]" + addParas, Class=Class)
    # X^2(1)
    h[1, i] = feval(Methods[method], "r[1][:, i]" + addParas, Class=Class)
    # GE(0.25)
    h[2, i] = feval(Methods[method], "r[2][:, i]" + addParas, Class=Class)
    # P(5)
    h[3, i] = feval(Methods[method], "r[3][:, i]" + addParas, Class=Class)
    # Exp(1)
    h[4, i] = feval(Methods[method], "r[4][:, i]" + addParas, Class=Class)
    # U(0,1)
    h[5, i] = feval(Methods[method], "r[5][:, i]" + addParas, Class=Class)

h = np.mean(h, axis=1)
print(method)
for i in h:
    print("{:.4f}".format(i))
