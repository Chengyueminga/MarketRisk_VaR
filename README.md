
# Monte Carlo VaR Simulation with Basel 3 Backtesting
This repository presents a self-initiated, educational market risk modeling project focusing on Monte Carlo Simulation of VaR and regulatory backtesting in Basel 3 requirement.

## Modeling Approaches
This project presents two distinct methods for estimating 1-day 99% Value-at-Risk (VaR) using Monte Carlo simulation, followed by regulatory backtesting under the Basel III framework. Both methods use the same portfolio input but differ in modeling assumptions and data generation logic.

## Modeling Approach 1:Historical Return-based Monte Carlo VaR Simulation

This approach directly simulates portfolio returns based on historical data. It assumes that the portfolio’s return follows a standard distribution, with the mean and volatility estimated from historical observations. Monte Carlo simulation is then performed using this distribution to estimate Value-at-Risk (VaR).

- **Input**: Historical daily returns of the portfolio.  
  Asset-level data is sourced from [Yahoo Finance](https://finance.yahoo.com/) via the `yfinance` Python package.
- **Assumption**: Returns are assumed to be independent and identically distributed (IID), and follow a normal distribution with historical mean and volatility.
- **Output**: Simulated 1-day return distribution via Monte Carlo; the 1st percentile represents the 99% Value-at-Risk (VaR).
- **Backtesting**: Based on the Basel III “traffic light” framework over 250 trading days. Realized VaR exceptions are counted and evaluated according to [BCBS22](https://www.bis.org/publ/bcbs22.pdf).

## Modeling Approach 2:Risk Factor-Based Monte Carlo VaR Simulation via Beta Sensitivity

This approach estimates the portfolio’s sensitivities (betas) to multiple market risk factors using linear regression. The beta-based logic is inspired by the Capital Asset Pricing Model (CAPM), as introduced in Risk Management and Financial Institutions by John C. Hull (4th Edition, Wiley, 2015). The CAPM structure is extended here to incorporate multiple factors in a simulation-based VaR framework.

## Project Summary
*Objective:*
  - Simulate 1-day 99% VaR for a hypothetical equity portfolio using Monte Carlo Simulation 

*Methodology:*
  - Simulate portfolio return paths under normally distributed assumptions.
  - Calculate portfolio-level VaR from simulated return paths.
  - Evaluate model performance using Basel III-style traffic light backtesting. Source: https://www.bis.org/publ/bcbs22.pdf

## Sample Outputs
- VaR distribution histogram
- Return vs. VaR plot
- Backtesting result with violation rate and zone

## View Full Analysis
Python Script:  
https://github.com/Chengyueminga/MarketRisk_VaR/blob/main/Basel3-VaR-Backtest_Monte-Carlo-Simulation.py

Jupyter Notebook:  
https://github.com/Chengyueminga/MarketRisk_VaR/blob/main/Basel3-VaR-Backtest_Monte-Carlo-Simulation.ipynb

## Source for VaR background
- Basel Committee on Banking Supervision (BCBS) — *Supervisory framework for the use of backtesting in conjunction with the internal models approach to market risk capital requirements*  
   https://www.bis.org/publ/bcbs22.pdf

- U.S. Office of the Comptroller of the Currency (OCC) — *Value-at-Risk Model Risk Control Guidelines*  
   https://www.occ.treas.gov/news-issuances/bulletins/1999/bulletin-1999-2.html



>  This project is intended for educational purposes and is based entirely on synthetic market data.

## License

This repository is publicly available for educational and demonstration purposes only.  
All rights reserved. Please do not reproduce or reuse the code without explicit permission.
