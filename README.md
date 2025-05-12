### Monte Carlo VaR Simulation
This repository presents a self-initiated, educational market risk modeling project focusing on Monte Carlo Simulation of VaR and regulatory backtesting in Basel 3 requirement.

## Project Summary
*Objective:* Simulate 1-day 99% VaR for a hypothetical equity portfolio using Monte Carlo Simulation 
*Methodology:*
  - Generate synthetic daily return scenarios assuming normal distribution.
  - Calculate portfolio-level VaR from simulated return paths.
  - Evaluate model performance using Basel III-style traffic light backtesting. Source: https://www.bis.org/publ/bcbs22.pdf

## Sample Outputs
- VaR distribution histogram
- Return vs. VaR plot
- Backtesting result with violation rate and zone

##View Full Analysis
Python Version:
https://github.com/Chengyueminga/MarketRisk_VaR/blob/main/Basel3-VaR-Backtest_Monte-Carlo-Simulation.py
Jupyter Notebook Version:
https://github.com/Chengyueminga/MarketRisk_VaR/blob/main/Basel3-VaR-Backtest_Monte-Carlo-Simulation.ipynb
