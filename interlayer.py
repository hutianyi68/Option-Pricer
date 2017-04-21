from core import *
import time

def calculate_vol_interlayer(r, q, stock, strike, t, price, option_type):
    start = time.time()
    if (r < 0) or (q < 0) or (stock <= 0) or (t <= 0) or (price <= 0):
        return "Invalid input !"
    result = calculate_vol(r, q, stock, strike, t, price, option_type)
    try:
        return_result = float(result)*100
    except Exception:
        return result
        pass
    return "The implied volatility is: " + str(round(return_result,4)) +\
 "%. And calculating time is " + str(round(time.time()-start,8))

def calculate_american_interlayer(r, s_0, k, t, sigma, n, option_type):
    start = time.time()
    if (r < 0) or (s_0 <= 0) or (k <= 0) or (t <= 0) or (sigma <= 0) or (n <= 0):
        return "Invalid input !"
    if n > 5000:
        return "The maximum number of steps is 5000 !"
    result = american_option(r, s_0, k, t, sigma, n, option_type)
    return "The american option price is: " + str(round(result,6)) +\
 ". And calculating time is " + str(round(time.time()-start,8))

def calculate_european_interlayer(r, q, s_0, k, sigma, t, option_type):
    start = time.time()
    if (r < 0) or (s_0 <= 0) or (k <= 0) or (t <= 0) or (sigma <= 0) or (q < 0):
        return "Invalid input !"
    result = black_scholes_model(r, q, s_0, k, sigma, t, option_type)
    return "The european option price is: " + str(round(result,6)) +\
 ". And calculating time is " + str(round(time.time()-start,8))

def calculate_geo_asian_interlayer(r, s_0, k, t, sigma, n, option_type):
    start = time.time()
    if (r < 0) or (s_0 <= 0) or (k <= 0) or (t <= 0) or (sigma <= 0) or (n <= 0):
        return "Invalid input !"
    result = geometric_asian_option(r, s_0, k, t, sigma, n, option_type)
    return "The geometric asian option price is: " + str(round(result,6)) +\
 ". And calculating time is " + str(round(time.time()-start,8))

def calculate_geo_basket_interlayer(r, s1_0, s2_0, k, t, sigma1, sigma2, rho, option_type):
    start = time.time()
    if (r < 0) or (s1_0 <= 0) or (s2_0 <= 0) or (k <= 0) or (t <= 0) or (sigma1 <= 0) or (sigma2 <= 0) or (rho < -1) or (rho > 1):
        return "Invalid input !"
    result = geometric_basket_option(r, s1_0, s2_0, k, t, sigma1, sigma2, rho, option_type)
    return "The geometric basket price is: " + str(round(result,6)) +\
 ". And calculating time is " + str(round(time.time()-start,8))

def calculate_arith_asian_interlayer(r, s_0, K, t, sigma, n, option_type, mc_times, control_variate):
    start = time.time()
    if (r < 0) or (s_0 <= 0) or (K <= 0) or (t <= 0) or (sigma <= 0) or (n <= 0) or (mc_times <= 0):
        return "Invalid input !"
    if control_variate == "No Control Variate":
        control_variate_tf = False
    else:
        control_variate_tf = True
    result = arithmetic_asian_option(r, s_0, K, t, sigma, n, option_type, mc_times, control_variate_tf)
    return "The 95% confidence interval is: " + str(round(result[1],6)) + " to " + str(round(result[2],6)) +\
 ". And calculating time is " + str(round(time.time()-start,8))

def calculate_arith_basket_interlayer(r, s1_0, s2_0, k, t, sigma1, sigma2, rho, option_type, mc_times, control_variate):
    start = time.time()
    if (r < 0) or (s1_0 <= 0) or (s2_0 <= 0) or (k <= 0) or (t <= 0) or (sigma1 <= 0) or (sigma2 <= 0) or (rho < -1) or (rho > 1) or (mc_times <= 0):
        return "Invalid input !"
    if control_variate == "No Control Variate":
        control_variate_tf = False
    else:
        control_variate_tf = True
    result = arithmetic_basket_option(r, s1_0, s2_0, k, t, sigma1, sigma2, rho, option_type, mc_times, control_variate_tf)
    return "The 95% confidence interval is: " + str(round(result[1],6)) + " to " + str(round(result[2],6)) +\
 ". And calculating time is " + str(round(time.time()-start,8))