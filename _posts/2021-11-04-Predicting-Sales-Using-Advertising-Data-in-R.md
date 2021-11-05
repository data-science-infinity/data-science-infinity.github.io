---
layout: post
title: Predicting Sales Using Advertising Data in R
image: "/posts/slidebean_unsplash.jpg"
tags: [R, Multiple Linear Regression]
---
How can we predict how much increase in sales will we get after spending money on different types of ads? I am using Advertising data here to show this and any company can collect this kind of data maybe from different branches and andwer the question: For how much are we going to increase sales spending the curtain amount on ads or how much money do the company needs to spend on ads to increase sales by X amount of dollars? 

---

Let's import the data and take a look
```R
Advertising <- read.csv("Advertising.csv", head=TRUE)
head(Advertising)
```
###### OUTPUT:
![output](/img/posts/output.png "output")

#### We will be answering 7 questions
##### Question 1: Is there a relationship between advertising and sales?


We will then fit the multiple regression model to test the Ho hypothese. We use ***F-statistic*** for this, we can see on the output below that the p-value(2.2e-16=0.00000000000000022) corresponding to the F-statistic(570.3) is very close to zero, indicating clear evidence of a relationship between advertising and sales.
```R
attach(Advertising)
advertising_fit <- lm(sales ~ TV + radio + newspaper)
summary(advertising_fit)
```
###### OUTPUT 1:
![output1](/img/posts/output1.png "output1")

#### Question 2: How strong is the relationship?
To answer this question we will look at the above output and use ***R-squared statistic***(0.8972) to decide. The value of 0.8972 means that the predictors explain almost 90% of the variance in sales, so the relationship is strong.

#### Question 3: Which media contribute to sales?
To answer this question, we can examine the p-values associated with
each predictor’s ***t-statistic*** using again the same output above. As we can see, the p-values for TV and radio are low and equal to 2e-16. We can't state the same about the p-value for newspaper which is 0.86. This suggests that only TV and radio are related to sales. 

#### Question 4: How large is the effect of each medium on sales?
We need to look at the ***confidence intervals*** of the coefficients
```
confint(advertising_fit)
```
###### OUTPUT 2:
![output2](/img/posts/output2.png "output2")
