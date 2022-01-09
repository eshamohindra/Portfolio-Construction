#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import scipy.stats
from scipy.stats import norm
from scipy.optimize import minimize

def get_ffme_returns():
    """
    Load the Fama-French Dataset for the returns of the Top and Bottom Deciles by MarketCap
    """
    me_m = pd.read_csv("Portfolios_Formed_on_ME_monthly_EW.csv",
                header=0, index_col=0, parse_dates = True, na_values = 99.99)

    rets = me_m[['Lo 10', 'Hi 10']]
    rets.columns = ['SmallCap', 'LargeCap']
    rets = rets/100
    rets.index = pd.to_datetime(rets.index, format='%Y%m').to_period("M")
    return rets 

def get_hfi_returns():
    """
    Load and format the EDHEC Hedge Fund Index returns
    """

    hfi = pd.read_csv("edhec-hedgefundindices.csv",
                header=0, index_col=0, parse_dates = True)

    hfi = hfi/100
    hfi.index = hfi.index.to_period('M')
    return hfi

def semideviation(r):
    """
    Returns semideviation aka negative semideviation of r
    r must be a Series of DataFrame
    """
    is_negative = r < 0
    return r[is_negative].std(ddof=0)

def skewness(r):
    """
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # Use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3

def kurtosis(r):
    """
    Alternative to scipy.stats.skew()
    Computes the kurtosis of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # Use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4

def is_normal(r, level = 0.01):
    """
    Applies the Jarque-Bera test tp determine if a Series is normal or not
    Test is applied at the 1% level by default
    Returns True if the hypothesis of normality is accepted, False otherwise
    """
    statistic, p_value = scipy.stats.jarque_bera(r)
    return p_value > level
    

def var_historic(r, level=5):
    """
    VaR Historic

    #Check if r is an instance of DataFrame, return True
    #If true, then var_historic is called again and this time due to 
    #aggregate function, columns of r as passed which are of type Series
    #Now it enters elif part and returns percentile of each column of r	
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
	#VaR is a positive number, hence the - sign
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be Series or DataFrame")

def var_gaussian(r, level=5, modified=False):
    """
    Returns the Parametric Gaussian VaR of a Series or DataFrame
    If "modified" is True, then the modified VaR is returned,
    using the Cornish-Fisher modification
    """
    
    #Compute the Z-Scoare assuming it was Gaussian(Normal)
    z = norm.ppf(level/100)
    
    if modified:
        #Modify the Z Score based on observed skewness and kurtosis
        
        s = skewness(r)
        k = kurtosis(r)
        z = (z + 
                (z**2 - 1)*s/6 +
                (z**3 - 3*z)*(k-3)/24 - 
                (2*z**3 - 5*z)*(s**2)/36
            )
    return -(r.mean() + z*r.std(ddof=0))

def cvar_historic(r, level=5):
    """
    CVaR of Series or DataFrame
    """
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be Series or DataFrame")

def drawdown(return_series: pd.Series):
    """
    Takes a time series of asset returns
    Computes and returns a DataFrame that contains:
    The Wealth Index
    Previous Peaks
    Percent drawdowns
    """
    
    wealth_index = 1000 * (1+return_series).cumprod()
    previous_peak = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peak)/previous_peak
    
    return pd.DataFrame({
        "Wealth" : wealth_index,
        "Peaks" : previous_peak,
        "Drawdown" : drawdowns
    })

def annualize_vol(r, periods_per_year):
    """
    Annualizes the vol of a set of returns.
    We should infer the periods per year
    """
    return r.std()*(periods_per_year**0.5)

def annualize_rets(r, periods_per_year):
    """
    Annualizes set of returns
    We should infer the periods
    """
    compounded_growth = (1+r).prod()
    
    #Returns number of rows
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1

def sharpe_ratio(r, riskfree_rate, periods_per_year):
    """
    Computes the annualized sharpe ratio of a set of returns
    """
    #Convert the annual riskfree rate to per period
    rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret/ann_vol

def portfolio_return(weights, returns):
    """
    Weight -> Returns
    """
    return weights.T @ returns

def portfolio_vol(weights, covmat):
    """
    Variance of the portfolio
    Weights -> Vol
    """
    return (weights.T @ covmat @ weights)**0.5

def plot_ef2(n_points, er, cov, style=".-"):
    """
    Plots the 2-asset efficient frontier
    """
    if er.shape[0] != 2 or er.shape[0]!=2:
        raise ValueError("plot_ef2 can only plot 2-asset frontiers")
    
    weights = [np.array([w, 1-w]) for w in np.linspace(0, 1, n_points)]
    rets = [portfolio_return(w, er) for w in weights]
    vol = [portfolio_vol(w, cov) for w in weights]
    
    ef = pd.DataFrame({
        "Returns" : rets,
        "Volatility" : vol
    })
    return ef.plot.line(x="Volatility", y="Returns", style=style)

def minimize_vol(target_return, er, cov):
    """
    Returns weights of a portfolio having 
    minimum volatility for given target return
    target_ret -> Weights
    
    Optimization requires: 
    Objective function - 
    Constraints - Weights should not be more than 1 and should not be negative. 
                  Return generated from set of weights is the return of target  
    Initial Guess - Equal weights to each asset
    """
    #Number of assets
    n = er.shape[0]
    
    #====== Initial guess of equal weights 
    init_guess = np.repeat(1/n, n)
    
    #====== Constraints
    #Tuple of tuples: Ex: ((0.0,1.0), (0.0,1.0), (0.0,1.0)...)
    bounds = ((0.0, 1.0),)*n
    
    #We need to make sure that the return generated from set of weights is the return of target
    return_is_target = {
        #Equality constraint
        'type': 'eq',
        
        #Additional Arguments to be sent to the function
        'args': (er,), 
        
        #Function to check if return generated from set of weights = return of target
        # Satisfies the constraint if return value is 0
        
        #Below function can be written using Lambda Function
        # def target_is_met(weights, er):
        #        return target_return - portfolio_return(weights, er)
        
        'fun': lambda weights, er: target_return - portfolio_return(weights, er)
    }
    
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
        
    #======= Run optimiser to generate set of weights
    
    result_weights = minimize(portfolio_vol, 
                       init_guess,
                       args = (cov,),
                       method = "SLSQP",
                       options = {'disp': False},
                       constraints = (return_is_target, weights_sum_to_1),
                       bounds = bounds,
                      )
    return result_weights.x

def optimal_weights(n_points, er, cov):
    """
    Generates a list of weights to run to optimizer on to minimize the volatility
    Generate linearly spaced target returns between minimum and maximum returns in expected returns values
    """
    target_returns = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_rs, er, cov) for target_rs in target_returns]
    return weights

def max_sharpe_ratio(riskfree_rate, er, cov):
    """
    Returns the weights of the portfolio that gives you the maximum Sharpe ratio given the RFR, expected return and covariance matrix
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),)*n

    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    
    def neg_sharpe_ratio(weights, riskfree_rate, er, cov):
        """
        Returns negative of the Sharpe Ratio given weights
        """
        returns = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(returns - riskfree_rate)/vol
    
    #Maximize Sharpe Ratio = Minimize negative Sharpe Ratio
    result_weights = minimize(neg_sharpe_ratio, 
                       init_guess,
                       args = (riskfree_rate, er, cov,),
                       method = "SLSQP",
                       options = {'disp': False},
                       constraints = (weights_sum_to_1),
                       bounds = bounds,
                      )
    return result_weights.x

def plot_ef(n_points, er, cov, show_cml=False,style=".-", riskfree_rate=0):
    """
    Plots the N-asset efficient frontier
    """
    
    weights = optimal_weights(n_points, er, cov)
    
    rets = [portfolio_return(w, er) for w in weights]
    vol = [portfolio_vol(w, cov) for w in weights]
    
    ef = pd.DataFrame({
        "Returns" : rets,
        "Volatility" : vol
    })
    ax = ef.plot.line(x="Volatility", y="Returns", style=style)
    
    if show_cml:
        ax.set_xlim(left = 0)
        w_msr = max_sharpe_ratio(riskfree_rate, er, cov)
        r_msr = portfolio_return(w_msr, er)
        vol_msr = portfolio_vol(w_msr, cov)

        #Add Capital Market Line
        cml_x = [0, vol_msr]
        cml_y = [riskfree_rate, r_msr]

        ax.plot(cml_x, cml_y, color="green", marker="o", linestyle="dashed", markersize=12, linewidth=2)
        
    return ax
