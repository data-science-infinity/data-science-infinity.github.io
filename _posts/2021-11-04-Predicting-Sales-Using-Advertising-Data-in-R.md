---
layout: post
title: Predicting Sales Using Advertising Data in R
image: "/posts/slidebean_unsplash.jpg"
tags: [R, Multiple Linear Regression]
---
How can we predict how much increase in sales will we get after spending money on different types of ads? I am using Advertising data here to show this and any company can collect this kind of data maybe from different branches and andwer the question: For how much are we going to increase sales spending the curtain amount on ads or how much money do the company needs to spend on ads to increase sales by X amount of dollars? 

#### We will be answering 7 questions
##### Question 1: Is there a relationship between advertising sales and budget?

First, we import the data and take a look
```
Advertising<-read.csv("Advertising.csv",head=TRUE)
head(Advertising)
```
We will then fit the multiple regression model to test the Ho hypothese and loot at the output
```
attach(Advertising)
advertising_fit <- lm(sales ~ TV + radio + newspaper)
summary(advertising_fit)
```
###### OUTPUT:
![![ouput1](/img/posts/output1.png "output1")


