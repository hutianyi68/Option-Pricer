# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 10:18:27 2017

@author: Lyn Ye
"""
import numpy as np
import scipy.stats as stats
DEFAULT_BINOMINAL_TREE_STEP = 2000

def PriceByBSFormula(T, K, S0, r, sigma, OptType, q =None) :
    d1 = ((np.log(S0 / K) + (r - q) * T) / (sigma * np.sqrt(T))+ 0.5 * sigma * np.sqrt(T))
    d2 = d1 - (sigma * np.sqrt(T))
    if OptType == "CALL":
        return (S0 * np.exp(-q * T) * stats.norm.cdf(d1)- K * np.exp(-r * T) * stats.norm.cdf(d2))
    else: 
       return (- S0 * np.exp(-q * T) *stats.norm.cdf(-d1) + K * np.exp(-r * T) * stats.norm.cdf(-d2))

def PriceBySnell(T, K, S0, r, sigma, OptType, N = DEFAULT_BINOMINAL_TREE_STEP):
    val_num_steps = N or DEFAULT_BINOMINAL_TREE_STEP
    val_cache = {}

    dt = float(T) / val_num_steps
    up_fact= np.exp(sigma * (dt ** 0.5))
    down_fact = 1.0 / up_fact
    prob_up = (np.exp((r ) * dt) - down_fact) / (up_fact - down_fact)
    def spot_price(num_ups, num_downs):
        return (S0) * (up_fact ** (num_ups - num_downs))
    def node_value(n, num_ups, num_downs):
        value_cache_key = (n, num_ups, num_downs)
        if value_cache_key not in val_cache:
            spot = spot_price(num_ups, num_downs)
            if  OptType == "CALL":
                exer_profit = max(0, spot - S0)
            else:  
                exer_profit = max(0, S0 - spot)

            if n >= val_num_steps:
                val = exer_profit
            else:
                fv = prob_up * node_value(n + 1, num_ups + 1, num_downs) + \
                    (1 - prob_up) * node_value(n + 1, num_ups, num_downs + 1)
                pv = np.exp(r * dt) * fv
                val = max(pv, exer_profit)
            val_cache[value_cache_key] = val
        return val_cache[value_cache_key]

    val = node_value(0, 0, 0)
    return val

def PriceByClosedFormulaForGmtrAsian(T, K, N,S0, r, sigma, OptType):
    a = S0 * np.exp(-r * T) * np.exp((N + 1) * T) / (2 * N) *\
               (r + (sigma * sigma / 2) * ((2 * N + 1) / (3 * N) - 1));
    b = sigma * np.sqrt((N + 1) * (2 * N + 1) / 6 / (N * N));
    return PriceByBSFormula(a, K, T, b, r)
    
    
def PriceByClosedFormulaForGmtBasket(T, K, S1, S2, r, sigma1, sigma2, rho, OptType):
    sigma_b = (sigma1 * sigma1 +sigma2 * sigma2 +2 *rho * sigma1 * sigma2) / 2
    niu_b =( r - (sigma1 * sigma1 + sigma2 * sigma2) / 4 + sigma_b * sigma_b / 2)
    S0_b = np.sqrt(S1 * S2)
    return PriceByBSFormula(S0_b, K, T, sigma_b, niu_b, 0)


def PriceByMCSimulationForArthmAsian(T, K, N,S0, r, sigma, OptType, NumPath, isGmtrCon =0):
    dT = T/N
    sqrtdT = np.sqrt(dT)
    deltaValConst = np.exp((r - 0.5 * sigma * sigma) * dT)
    sample = np.random.normal(size=(NumPath,N))#(3,2)
    sample = np.exp(sample * sqrtdT * sigma) * deltaValConst * S0
    if OptType == "CALL":
        sample = np.maximum( sample -K, 0)
    else:
        sample = np.maximum( K - sample , 0) 
    if isGmtrCon ==0:                  
        sampleNumPath = sample.sum(1)/N                          
        mean_sampleNumPath = sampleNumPath.mean()
        std_sampleNumPath = sampleNumPath.std()
        
        OptVal =  mean_sampleNumPath
        LCI = mean_sampleNumPath - 1.96 * std_sampleNumPath
        RCI = mean_sampleNumPath + 1.96 * std_sampleNumPath
    else: 
        GmtrConVal = PriceByClosedFormulaForGmtrAsian(T, K, N,S0, r, sigma, OptType)                  
        sampleNumPath = sample.sum(1)/N
        sampleNumPathGmtr= sample.prod(1)**(1/N)
        sampleNumPathErrRed = sampleNumPath - sampleNumPathGmtr + GmtrConVal
        mean_sampleErrRed = sampleNumPathErrRed.mean()
        std_sampleErrRed = sampleNumPathErrRed.std()
        
        OptVal =  mean_sampleErrRed
        LCI = mean_sampleErrRed - 1.96 * std_sampleErrRed
        RCI = mean_sampleErrRed + 1.96 * std_sampleErrRed
        
    return OptVal, LCI, RCI


def PriceByMCSimulationForArthmBasket(T, K, S1, S2, r, sigma1, sigma2, rho, OptType,NumPath, isGmtrCon =0):
    sqrtT = np.sqrt(T)
    deltaValConst1 = np.exp((r - 0.5 * sigma1 * sigma1) * T)
    deltaValConst2 = np.exp((r - 0.5 * sigma1 * sigma1) * T)
    sample_1 = np.random.normal(size=(NumPath,1))#(3,1)
    sample_ = np.random.normal(size=(NumPath,1))
    sample_2 = sample_1 * rho + np.sqrt(1 - rho * rho) * sample_
    sample_1 = np.exp(sample_1 * sqrtT * sigma1) * deltaValConst1 * S1
    sample_2 = np.exp(sample_2 * sqrtT * sigma2) * deltaValConst2 * S1
    if OptType == "CALL":
        sample_1 = np.maximum( sample_1 -K, 0)
        sample_2 = np.maximum( sample_2 -K, 0)
    else:
        sample_1 = np.maximum( K - sample_1, 0)
        sample_2 = np.maximum( K - sample_2 , 0)
    sample = (sample_1 +sample_2)/2                
    if isGmtrCon == 0: 
        mean_sample = sample.mean()
        std_sample = sample.std()
        OptVal =  mean_sample
        LCI = mean_sample - 1.96 * std_sample
        RCI = mean_sample + 1.96 * std_sample
    else: 
        GmtrConVal = PriceByClosedFormulaForGmtBasket(T, K, S1, S2, r, sigma1, sigma2, rho, OptType,NumPath)
        sampleGmtr = sample_1 * sample_2 ** (1/2)
        sampleErrRed = sample - sampleGmtr + GmtrConVal
        mean_sampleErrRed = sampleErrRed.mean()
        std_sampleErrRed = sampleErrRed.std()
        
        OptVal =  mean_sampleErrRed
        LCI = mean_sampleErrRed - 1.96 * std_sampleErrRed
        RCI = mean_sampleErrRed + 1.96 * std_sampleErrRed
        
    return OptVal, LCI, RCI

    
    
    
    
    
    
    
    
    