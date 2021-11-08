---
layout: post
title: Time Series Modelling and Forecasting in R
image: "/posts/chris_liverani_unsplash.jpg"
tags: [R, Time Series, Forecasting]
---
We will be modeling data om average number of goals for English Premier League and forecasting for the future years

---
##### We will need the following packages and libraries

```r
library(forecast)
library(timeDate)
library(timeSeries)
library(fBasics)
library(fGarch)
```
##### Importing data and removing the fists columns which is just numbering the rows


```r
Premierleague_goals <- read.csv("Premierleague_goals.csv", header=FALSE, comment.char="#")[,-1]
head(Premierleague_goals)

```
![goals1](/img/posts/goals1.png "goals1")

#### View our data and its ***acf*** plots

```r
par(mfrow=c(3,2))
ts.plot(Premierleague_goals[,2])
acf(Premierleague_goals[,2])
ts.plot(Premierleague_goals[,3])
acf(Premierleague_goals[,3])
ts.plot(Premierleague_goals[,4])
acf(Premierleague_goals[,4])
```
![goals2](/img/posts/goals2.png "goals2")

#### As the ***ACF*** for Premierleague_goals[,4] data is ***decaying linearly***, we have evidence of a ***unit root*** so we need to ***difference***. The differenced data seem to exhibit more stationary features. 

```r
ts.plot(diff(Premierleague_goals[,4]))
acf(diff(Premierleague_goals[,4]))
```

![goals3](/img/posts/goals3.png "goals3")

#### We will also remove the first 4 observations as they seem a bit out-of-line with the rest of the data and look at the plots

```r
x=diff(Premierleague_goals[,4])
x=x[-(1:4)]
tsdisplay(x)
```
![goals4](/img/posts/goals4.png "goals4")

#### Since the ***ACF*** and ***PACF*** seems to suggest some spikes of order p = 3, q = 3, we might like to consider those values together with the smaller models. We use ***arima*** command which uses ***Maximum Likelihood Estimation*** to calculate the coefficients

```r
fit1=arima(x,c(1,0,1))
fit2=arima(x,c(2,0,1))
fit3=arima(x,c(3,0,1))
fit4=arima(x,c(3,0,0))
fit5=arima(x,c(3,0,2))
fit6=arima(x,c(2,0,2))
fit7=arima(x,c(1,0,2))
fit8=arima(x,c(0,0,2))
fit9=arima(x,c(0,0,3))
fit10=arima(x,c(1,0,3))
```

#### We are choosing the model with the smallest ***AIC***. It is observed for model 10: fit10=arima(x,c(1,0,3))

![goals5](/img/posts/goals5.png "goals5")

#### We can perform a model diagnostics for fit10, by using ***tsdiag*** which uses the old ***Ljung-Box test statistics***

```r
tsdiag(fit10)
```
![goals6](/img/posts/goals6.png "goals6")

#### But the best goodness of fit test is ***Weighted Monti test*** as it is the most powerful


```r
install.packages('WeightedPortTest')
library('WeightedPortTest')
Weighted.Box.test(residuals(fit10), lag = 10, 
                  type = "Monti",
                  fitdf = 0, weighted = TRUE)
```
![goals7](/img/posts/goals7.png "goals7")

#### Now when we have the best model, we can ***forecast*** for let's say 4 future Premier League seasons

```r
forecast(fit10, h=4)
plot(forecast(fit10, h=4))
```r
![goals8](/img/posts/goals8.png "goals8")
![goals9](/img/posts/goals9.png "goals9")






