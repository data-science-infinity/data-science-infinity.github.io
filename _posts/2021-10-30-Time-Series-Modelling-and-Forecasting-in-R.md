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

#### As the ACF for Premierleague_goals[,4] data is ***decaying linearly***, we have evidence of a ***unit root*** so we need to ***difference***. The differenced data seem to exhibit more stationary features. 

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
