---
layout: post
title: Time Series Modelling and Forecasting in R
image: "/posts/chris_liverani_unsplash.jpg"
tags: [R, Time Series, Forecasting]
---
We will be modeling data om average number of goals for English Premier League and forecasting for the future years

---
We will need the following packages and libraries

```r
library(forecast)
library(timeDate)
library(timeSeries)
library(fBasics)
library(fGarch)
```
Importing data and removing the fists columns which is just numbering the rows


```r
Premierleague_goals <- read.csv("Premierleague_goals.csv", header=FALSE, comment.char="#")[,-1]
head(Premierleague_goals)

```
![goals1](/img/posts/goals1.png "goals1")
