import numpy as np
from numba import jit
from scipy.stats import norm

def black_scholes_model(r, q, s_0, k, sigma, t, option_type):
    d_1 = (np.log(s_0/k)+(r-q+0.5*np.square(sigma))*t)/(np.sqrt(t)*sigma)
    d_2 = (np.log(s_0/k)+(r-q-0.5*np.square(sigma))*t)/(np.sqrt(t)*sigma)
    if option_type == "Call":
        return s_0*np.exp(-q*t)*norm.cdf(d_1) - k*np.exp(-r*t)*norm.cdf(d_2)
    if option_type == "Put":
        return k*np.exp(-r*t)*norm.cdf(-d_2) - s_0*np.exp(-q*t)*norm.cdf(-d_1)

def calculate_vol_detail(option_type, sigma, stock, strike, price, q, r, t, accuracy, time):
    if (option_type == "Put") and (price < (strike*np.exp(-r*t) - stock*np.exp(-q*t))):
        return "Price is out of the valid range !"
    if (option_type == "Call") and (price < (- strike*np.exp(-r*t) + stock*np.exp(-q*t))):
        return "Price is out of the valid range !"
    if (sigma < 0) or (stock <= 0) or (strike <= 0 ):
        return "NaN"
    d_1 = (np.log(stock/strike)+(r-q+0.5*np.power(sigma,2))*t)/(sigma*np.sqrt(t))
    d_2 = (np.log(stock/strike)+(r-q-0.5*np.power(sigma,2))*t)/(sigma*np.sqrt(t))
    if option_type == 'Call':
        epsilon = -(stock*np.exp(-q*t)*norm.cdf(d_1)-strike*np.exp(-r*t)*norm.cdf(d_2)-price)/(stock*np.exp(-q*t)*np.sqrt(t)*norm.pdf(d_1))
    elif option_type == 'Put':
        epsilon = -(-stock*np.exp(-q*t)*norm.cdf(-d_1)+strike*np.exp(-r*t)*norm.cdf(-d_2)-price)/(stock*np.exp(-q*t)*np.sqrt(t)*norm.pdf(d_1))
    else:
        return "NaN"
    if time > 1000:
        return "Too many recursions and may cause stack-over-flow !"
    if abs(epsilon) < accuracy:
        return sigma
    else:
        return calculate_vol_detail(option_type, sigma+epsilon, stock, strike, price, q, r, t, accuracy, time+1)
 
def calculate_vol(r, q, stock, strike, t, price, option_type):
    accuracy = 0.001
    time = 0
    sigma = np.sqrt(2*abs((np.log(stock/strike)+(r-q)*t)/(t)))
    try:
        result = calculate_vol_detail(option_type, sigma, stock, strike, price, q, r, t, accuracy, time+1)
    except:
        result = "NaN"
    return result

def american_option(r, s_0, k, t, sigma, n, option_type):
    up = np.exp(sigma*np.sqrt(t/n))
    down = np.exp(-sigma*np.sqrt(t/n))
    up_prob = (np.exp(r*t/n)-down)/(up-down)
    down_prob = (up-np.exp(r*t/n))/(up-down)
    value = {}
    for i in range(n, -1, -1):
        for j in range(i, -1, -1):
            if option_type == "Call":
                if i == n:
                    value[(j, i)] = max(s_0*np.power(down,j)*np.power(up,i-j)-k,0)
                else:
                    value[(j, i)] = max(max(s_0*np.power(down,j)*np.power(up,i-j)-k,0), np.exp(-r*t/n)*\
                         (up_prob*value[(j,i+1)]+down_prob*value[(j+1,i+1)]))
            if option_type == "Put":
                if i == n:
                    value[(j, i)] = max(k-s_0*np.power(up,i-j)*np.power(down,j),0)
                else:
                    value[(j, i)] = max(max(k-s_0*np.power(up,i-j)*np.power(down,j),0), np.exp(-r*t/n)*\
                         (up_prob*value[(j,i+1)]+down_prob*value[(j+1,i+1)]))
    return value[(0,0)]

def geometric_asian_option(r, s_0, k, t, sigma, n, option_type):
    sigma_bar = sigma*np.sqrt((n+1)*(2*n+1)/(6*n**2))
    mu_bar = (r-0.5*sigma**2)*(n+1)/(2*n)+0.5*sigma_bar**2
    d_1 = (np.log(s_0/k)+(mu_bar+0.5*sigma_bar**2)*t)/(sigma_bar*np.sqrt(t))
    d_2 = (np.log(s_0/k)+(mu_bar-0.5*sigma_bar**2)*t)/(sigma_bar*np.sqrt(t))
    if option_type == "Call":
        return np.exp(-r*t)*(s_0*np.exp(mu_bar*t)*norm.cdf(d_1)-k*norm.cdf(d_2))
    if option_type == "Put":
        return np.exp(-r*t)*(k*norm.cdf(-d_2)-s_0*np.exp(mu_bar*t)*norm.cdf(-d_1))
    
@jit
def arithmetic_asian_option(r, s_0, K, t, sigma, n, option_type, mc_times, control_variate):
    df = np.exp(-r*t)
    drift = (r-0.5*np.square(sigma))*t/n
    squared_time = np.sqrt(t/n)
    stock_matrix = np.zeros((mc_times,n))
    for i in range(0, mc_times):
        stock_matrix[i][0] = s_0
        for j in range(0, n):
            if j == 0:
                stock_matrix[i][j] = s_0*np.exp(drift+sigma*squared_time*np.random.standard_normal())
            else:
                stock_matrix[i][j] = stock_matrix[i][j-1]*np.exp(drift+sigma*squared_time*np.random.standard_normal())
    stock_matrix_arith = stock_matrix.mean(1)
    if option_type == "Call":
        option_value = np.maximum(stock_matrix_arith - K, 0)*df
    else:
        option_value = np.maximum(K - stock_matrix_arith, 0)*df
    if control_variate == False:
        result_mean = option_value.mean()
        result_stderr = option_value.std()/np.sqrt(mc_times)
    else:
        geo_mean = geometric_asian_option(r, s_0, K, t, sigma, n, option_type)
        stock_matrix_geo = stock_matrix.prod(1)**(1/n)
        if option_type == "Call":
            option_value_geo = np.maximum(stock_matrix_geo - K, 0)*df
        else:
            option_value_geo = np.maximum(K - stock_matrix_geo, 0)*df
        theta = np.cov(option_value, option_value_geo)[0,1]/option_value_geo.var()
        option_value_with_cv = option_value + theta*(geo_mean - option_value_geo)
        result_mean = option_value_with_cv.mean()
        result_stderr = option_value_with_cv.std()/np.sqrt(mc_times)
    return result_mean, result_mean-1.96*result_stderr, result_mean+1.96*result_stderr
        
    
def geometric_basket_option(r, s1_0, s2_0, k, t, sigma1, sigma2, rho, option_type):
    sigma_bar = np.sqrt(sigma1**2+sigma2**2+2*sigma1*sigma2*rho)/2
    mu_bar = r - 0.5*(sigma1**2+sigma2**2)/2 + 0.5*sigma_bar**2
    d_1 = (np.log(np.sqrt(s1_0*s2_0)/k) + (mu_bar+0.5*sigma_bar**2)*t)/(sigma_bar*np.sqrt(t))
    d_2 = (np.log(np.sqrt(s1_0*s2_0)/k) + (mu_bar-0.5*sigma_bar**2)*t)/(sigma_bar*np.sqrt(t))
    if option_type == "Call":
        return np.exp(-r*t)*(np.sqrt(s1_0*s2_0)*np.exp(mu_bar*t)*norm.cdf(d_1)-k*norm.cdf(d_2))
    if option_type == "Put":
        return np.exp(-r*t)*(k*norm.cdf(-d_2)-np.sqrt(s1_0*s2_0)*np.exp(mu_bar*t)*norm.cdf(-d_1))
    
@jit
def arithmetic_basket_option(r, s1_0, s2_0, k, t, sigma1, sigma2, rho, option_type, mc_times, control_variate):
    geo_mean = geometric_basket_option(r, s1_0, s2_0, k, t, sigma1, sigma2, rho, option_type)
    normal_random_1 = np.random.standard_normal(size = mc_times)
    normal_random_temp = np.random.standard_normal(size = mc_times)
    normal_random_2 = rho*normal_random_1 + np.sqrt(1-rho**2)*normal_random_temp
    stock1 = s1_0*np.exp((r-0.5*sigma1**2)*t+sigma1*np.sqrt(t)*normal_random_1)
    stock2 = s2_0*np.exp((r-0.5*sigma2**2)*t+sigma2*np.sqrt(t)*normal_random_2)
    stock_matrix = np.column_stack((stock1, stock2))
    stock_arith = stock_matrix.mean(1)
    stock_geom = stock_matrix.prod(1)**(1/2)
    if option_type == "Call":
        option_arith = np.maximum(stock_arith-k,0)*np.exp(-r*t)
        option_geom = np.maximum(stock_geom-k,0)*np.exp(-r*t)
    else:
        option_arith = np.maximum(k-stock_arith,0)*np.exp(-r*t)
        option_geom = np.maximum(k-stock_geom,0)*np.exp(-r*t)
    if control_variate == False:
        result_mean = option_arith.mean()
        result_std = option_arith.std()/np.sqrt(mc_times)
    else:
        cov = np.cov(option_arith, option_geom)[0,1]
        theta = cov/option_geom.var()
        option_arith_cv = option_arith + theta*(geo_mean - option_geom)
        result_mean = option_arith_cv.mean()
        result_std = option_arith_cv.std()/np.sqrt(mc_times)
    return result_mean, result_mean-1.96*result_std, result_mean+1.96*result_std