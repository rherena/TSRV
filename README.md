# TSRV
Time Series Analysis on Relative Value ETFs

In this Package we are using datascience / ML models to forecast relative returns for sector ETF's.

In particular we focus on Sector ETF performance relative to SPY (the broad market)

We have two notebooks and a folder structure to hold all our EDA, model testing/validation, & back test results:

	- EDA_s0 holds our introductory scatter plots, shows how we compute all features*
		- We show cointegration using ADF on relative returns
		- Plot our features for visual confirmation/priming of modeling relationship
		- Show how correlation can be used to split our ETF's into seperate datasets

	- ModelFit_s2 holds our modeling process from start to finish
		- This includes Random Forest/Logistic Regression Models
		- Demo of using feature importance for financial markets research * 
		- Combinatorial Backtesting Method 
		- Walk Forward Backtesting Method

Sample Plots from our Analysis:

![Combinatorial Backetest Results 3m Sharpes](plots/FinalLogisticRegressionPlot.png "CombBackTest")




Walk Forward Tabular Results:

| Tables        | Are           | Cool  |
| ------------- |:-------------:| -----:|
| col 3 is      | right-aligned | $1600 |
| col 2 is      | centered      |   $12 |
| zebra stripes | are neat      |    $1 |


* We used the ta package heavily for creation of technical features: 
	[TA](https://github.com/bukosabino/ta)
* We use Methods in our feature construction & modeling from Marcos Lopez de Prado Advances in Financial Machine Learning (2018)
