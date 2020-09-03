# N-Tuple patterns to forecast market movements

A. G. Malliaris, in his paper "N-tuple S&P patterns across decades, 1950-2011", investigates the impact that past patterns have when forecasting future market movements of stock prices.

Specifically: can you leverage how the stock moved in the last 7 days to forecast how it will move later on?

This codebase tries to investigate that claim. I coded it on the data of S&P500 from 1950 to 2011 as Malliaris did, of the Italian index Comit Generale from 1973 to 2016, and on the American Express stock price from 1972 to 2016. 

To forecast, I tested a set of different tools (Random Walk, Decision Tree, Random Forest a Neural Network).

In the end, a vanilla Logistic Regression worked best, correctly predicting 63.8% of the times the price movement (up from Malliaris' 56%). The other interesting insight is that increasing the variable count of past movement, from the 3rd day onwards, the performance decreases for all models, hinting 1. overfit, 2. past irrelevance.

The code is provided as is. Use it as your own inspiration. Remeber to cite it in your work. MIT license (2017).
