
# Car Servicing Company Revenue Forecasting

## Context
The context of this project is to forecast the revenue of the car servicing company
using machine learning algorithms that play an important role in optimizing forecasting 
process in leveraging the data & addressing issues such as service time, stock maintenance, marketing, 
and discounts, etc and also plays a major role in decision-making 
operations in the areas corresponding to establishing the country wide network by increaseing the number of workshops.

## Aim and Objective
The aim of this project is to investigate the various revenue forecasting methods executed in financial area and evaluate the performance of the chosen machine learning
algorithms to find out the best suitable and efficient model for the chosen data set.

#### Objective
- To understand the efficient machine learning techniques for forecasting the revenue.
- To evaluate the performance of the selected machine learning algorithms by comparing the degree of prediction success rate. 

## Technical Aspects
This project is divided in two parts:
- Exploratory Data Analysis (EDA)
- SARIMAX Time Series Algorithm Implementation


### Exploratory Data Analysis (EDA)
We have the following files provided from as the part of the project:
1. customer_data.csv - Contained all the customer information.
2. invoice_data.csv & invoice_data1.csv - Contained all the information regarding the invoice and the revenue. 
3. invoice_plant_joined.csv - Contained all the information regarding the different plants established
4. Plant_Master.csv - Master file of the all thw workshops
5. final_invoice.csv - Created by combining the invoice_data.csv and invoice_data1.csv


#### Started with Univariate Analysis of the data:
- District
- District with respect to States
- City

#### Bi-Variate Analysis
- Make (Car Company) VS Total revenue genearted by brands
- Top 5 Car Models VS Total revenue generated by Cars 
and more bi-variate analyis were carried out..(refer EDA1.ipynb)

The most important analysis included the Service Time. The Service time was calculated using Job Card Time (In Time) parameter and the Gate Pass Time (Out Time). Various Bi-Variate Analysis was implemented using the service time parameter w.r.to revenue forecasting.

### SARIMAX Time Series Algorithm Implementation
Seasonal Autoregressive Integrated Moving Average, SARIMA or Seasonal ARIMA, is an extension of ARIMA that explicitly supports univariate time series data with a seasonal component.

It adds three new hyperparameters to specify the autoregression (AR), differencing (I) and moving average (MA) for the seasonal component of the series, as well as an additional parameter for the period of the seasonality.

#### Akaike information criterion (AIC)
Akaike information criterion (AIC) is a single number score that can be used to determine which of multiple models is most likely to be the best model for a given dataset. It estimates models relatively, meaning that AIC scores are only useful in comparison with other AIC scores for the same dataset. A lower AIC score is better.

## INSTALLATION
The Code is written in Python 3.7. If you don't have Python installed you can find it [there](https://www.python.org/downloads/)
. If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. 
To install the required packages and libraries, run this command in the project directory after [cloning](https://www.howtogeek.com/451360/how-to-clone-a-github-repository/) the repository:

    pip install -r requirements.

## Conclusion and Future Work
- Implementation of SARIMAX Time Series Algorithm for the Revenue Forecasting is successful. 
- The model can predict the revenue for the upcoming months entered by the end user.

- Such Implementation can be used the business-operation team.

#### Future Work
- Various other Time Series Model can be implemented for Revenue Forecasting such as FBProphet Time Series Algorithm.
- End-to-end Implementation of the project can also be done.
## Lessons Learned

SARIMAX Time Series Algorithm was new to me at the time of implementation of Revenue Forecasting. Eventually reffered various documentations and learned the art of implementing Time Series Projects

Also calculation the AIC Value was a bit difficult as multiple models needed to executed simultaneously.
## Contributors
Implementing a Machine Learning Time Series Model isn`t easy.

Along with me, there are 2 other contributors as well:
    
1. Viraj Trivedi https://github.com/VirajTrivedi17
2. Krina Shah 


## References

Thank You Rowhit Swami showing the way of creating the README File

https://github.com/rowhitswami

https://medium.com/analytics-vidhya/how-to-guide-on-exploratory-data-analysis-for-time-series-data-34250ff1d04f

https://towardsdatascience.com/introduction-to-aic-akaike-information-criterion-9c9ba1c96ced
