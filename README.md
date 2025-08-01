
# Monte Carlo VaR Simulation with Basel 3 Backtesting

[![GitHub stars](https://img.shields.io/github/stars/Chengyueminga/MarketRisk_VaR?style=social)](https://github.com/Chengyueminga/MarketRisk_VaR/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Chengyueminga/MarketRisk_VaR?style=social)](https://github.com/Chengyueminga/MarketRisk_VaR/network/members)

>  If you find this project helpful, please consider ⭐️ starring or 🍴 forking it to support and share with others!

**Disclaimer**: This repository presents an independent, educational research project focused on market risk modeling. It explores Monte Carlo-based Value-at-Risk (VaR) estimation and regulatory backtesting within the public Basel III framework. All data and modeling methodologies are based on public sources (e.g., Yahoo Finance, standard academic references), and no proprietary models, internal systems, or institutional views are used or implied.

## SSRN Paper Link: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5284626

## Modeling Approaches
This project presents two distinct methods for estimating 1-day 99% Value-at-Risk (VaR) using Monte Carlo simulation, followed by regulatory backtesting under the Basel III framework. Both methods use the same portfolio input but differ in modeling assumptions and data generation logic.

## Modeling Approach 1:Historical Return-based Monte Carlo VaR Simulation

This approach directly simulates portfolio returns based on historical data. It assumes that the portfolio’s return follows a standard distribution, with the mean and volatility estimated from historical observations. Monte Carlo simulation is then performed using this distribution to estimate Value-at-Risk (VaR).

- **Input**: Historical daily returns of the portfolio.  
  Asset-level data is sourced from [Yahoo Finance](https://finance.yahoo.com/) via the `yfinance` Python package.
- **Assumption**: Returns are assumed to be independent and identically distributed (IID), and follow a normal distribution with historical mean and volatility.
- **Output**: Simulated 1-day return distribution via Monte Carlo; the 1st percentile represents the 99% Value-at-Risk (VaR).
- **Backtesting**: Based on the Basel III “traffic light” framework over 250 trading days. Realized VaR exceptions are counted and evaluated according to [BCBS22](https://www.bis.org/publ/bcbs22.pdf).
- **Rolling Backtesting**: To better reflect time-varying market volatility, a rolling window approach is applied. For each day in the test period, a 1-day 99% VaR is estimated using the most recent 250 trading days as the training window. This dynamic framework captures evolving risk conditions more realistically compared to a static backtest, helping identify periods of model underperformance.
  
[Open Notebook](https://github.com/Chengyueminga/MarketRisk_VaR/blob/main/Basel3-VaR-Backtest_Monte-Carlo-Simulation.ipynb)

## Modeling Approach 2:Risk Factor-Based Monte Carlo VaR Simulation via Beta Sensitivity

This approach estimates the portfolio’s sensitivities (betas) to multiple market risk factors using linear regression. The beta-based logic is inspired by the Capital Asset Pricing Model (CAPM), as introduced in Risk Management and Financial Institutions by John C. Hull (4th Edition, Wiley, 2015). The CAPM structure is extended here to incorporate multiple factors in a simulation-based VaR framework.

The model incorporates the following risk factors:

- **SPY**: Broad U.S. equity market index  
- **XLK**: Technology sector ETF  
- **TLT**: Long-duration Treasury bond ETF  
- **MTUM**: Momentum ETF  
- **^VIX**: Market volatility index  
- **VLUE**: Value-style ETF  
- **IWF**: Growth-style ETF

These factors collectively represent market, sector, style, and macroeconomic exposures relevant to a diversified, tech-centric portfolio.

- **Input**: Historical daily returns of the selected risk factors (SPY, XLK, TLT, MTUM, ^VIX, VLUE, IWF).  
  Data is sourced from [Yahoo Finance](https://finance.yahoo.com/) via the `yfinance` Python package.

- **Assumption**:  
  - The relationship between portfolio returns and risk factors is assumed to be linear and stable over the estimation window.  
  - Risk factor returns follow a multivariate normal distribution, calibrated to historical means and covariances.  
  - Portfolio sensitivities (betas) are constant and estimated using linear regression.  
  - Multicollinearity among factors is evaluated using a correlation matrix and Variance Inflation Factor (VIF) diagnostics. In practice, multicollinearity is only considered a concern when statistically insignificant factors are retained.
  - Simulated risk factor returns are generated using Cholesky decomposition on the historical covariance matrix, ensuring that the simulated data preserves empirical correlations.
  - Other linear regression assumptions (e.g., residual independence, homoscedasticity) are not explicitly tested and are considered outside the scope of this simulation.
    
- **Output**: Portfolio returns are generated by applying estimated betas (via linear regression) to simulated risk factor returns, which follow a multivariate normal distribution.  
  The 1st percentile of the resulting return distribution represents the 99% Value-at-Risk (VaR).

- **Backtesting**: Based on the Basel III “traffic light” framework over 250 trading days. Realized VaR exceptions are counted and evaluated according to [BCBS22](https://www.bis.org/publ/bcbs22.pdf).

- **Rolling Backtesting**: To better reflect time-varying market volatility, a rolling window approach is applied. For each day in the test period, a 1-day 99% VaR is estimated using the most recent 250 trading days as the training window. This dynamic framework captures evolving risk conditions more realistically compared to a static backtest, helping identify periods of model underperformance.
 
[Open Notebook](https://github.com/Chengyueminga/MarketRisk_VaR/blob/main/Beta-Based%20Risk%20Factor%20VaR%20Simulation%20for%20Basel%20III%20Backtesting%20.ipynb)


## Conclusion and Future Work

This project demonstrates two practical approaches to Monte Carlo-based Value-at-Risk (VaR) estimation and regulatory backtesting using publicly available data. The factor-based model introduces a flexible framework for simulating risk under multi-dimensional exposures.

Future extensions may include:
- Time-series-based factor simulation (e.g., using GARCH models)
- Non-linear risk factor interactions
- Stress testing under extreme market events

## Acknowledgment of Collaboration

This project benefited from insightful input by H.Fang, who proposed the rolling backtesting structure. Their contribution helped align the simulation output with the Basel regulatory backtesting framework, ensuring that the Monte Carlo VaR models were evaluated under conditions consistent with real-world regulatory expectations.
Their expertise in stress testing and scenario-based capital modeling provided practical context for implementing a forward-looking evaluation of VaR model performance.

## License

This project is licensed under the MIT License. 
See the [LICENSE](LICENSE) file for full details.
