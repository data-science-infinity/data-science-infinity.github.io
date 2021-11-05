---
layout: post
title: Predicting Sales Using Advertising Data in R
image: "/posts/slidebean_unsplash.jpg"
tags: [R, Multiple Linear Regression]
---
How can we predict how much increase in sales will we get after spending money on different types of ads? I am using Advertising data here to show this and any company can collect this kind of data maybe from different branches and andwer the question: For how much are we going to increase sales spending the curtain amount on ads or how much money do the company needs to spend on ads to increase sales by X amount of dollars? 

---

Let's import the Advertising data and take a look. On the output below we can see that we have 3 predictors (TV, radio, newspaper) and 1 response variable - sales and the sample size is 200, data from two hundred cities.
```R
Advertising <- read.csv("Advertising.csv", head=TRUE)
head(Advertising)
```
###### OUTPUT:
![output](/img/posts/output.png "output")

#### We will be answering 7 questions
##### Question 1: Is there a relationship between advertising and sales?

To answer this question, we will then fit the multiple regression model to test the Ho hypothese. We use ***F-statistic*** for this, there is a relationship between the response and predictors when F-statistic is greater than 1. But how big is big? When the p-value is less than 0.05, we can conclude that there is a relationship. As we can see on the output1 below the p-value(2.2e-16=0.00000000000000022) corresponding to the F-statistic(570.3) is very close to zero, indicating clear evidence of a relationship between advertising and sales.
```R
attach(Advertising)
advertising_fit <- lm(sales ~ TV + radio + newspaper)
summary(advertising_fit)
```
###### OUTPUT 1:
![output1](/img/posts/output1.png "output1")

#### Question 2: How strong is the relationship?
We will look at the above output1 and use ***R-squared statistic***(0.8972) to decide. The value of 0.8972 means that the predictors explain almost 90% of the variance in sales, so the relationship is strong.

#### Question 3: Which media contribute to sales?
To answer this question, we need to examine the p-values associated with each predictor’s ***t-statistic*** using again the same output1 above. As we can see, the p-values for TV and radio are low and equal to 2e-16. We can't state the same about the p-value for newspaper which is 0.86. This suggests that only TV and radio are related to sales. 

#### Question 4: How large is the effect of each medium on sales?
We need to look at the ***confidence intervals*** of the coefficients. Looking at the output2 below we can see that for the Advertising data, the 95% confidence intervals are as follows: (0.043, 0.049) for TV (coefficient value=0.046), (0.172, 0.206) for radio (0.189), and (−0.013, 0.011) for newspaper (-0.001). The confidence intervals for TV and radio are narrow and far from zero, providing
evidence that these media are related to sales. But the interval for newspaper includes zero, indicating that the variable is not statistically significant given the values of TV and radio.
```R
confint(advertising_fit)
```
###### OUTPUT 2:
![output2](/img/posts/output2.png "output2")

We will examine this further to see if ***collinearity*** be the reason that the confidence interval associated with newspaper is so wide? From the output 3 we can see that the VIF scores are 1.005, 1.145, and 1.145 for TV, radio, and newspaper, suggesting no evidence of collinearity.

###### OUTPUT 3:
![output3](/img/posts/output3.png "output3")

In order to assess how large is the effect of each medium individually on sales, we can fit three separate ***simple linear regression models***. Results are shown on output4 and output5. There is evidence of an ***extremely strong association*** between TV and sales and between radio and sales. There is evidence of a ***mild association*** between newspaper and sales (output6), when the values of TV and radio are ignored.
```R
advertising_fit_TV <- lm(sales ~ TV)
advertising_fit_radio <- lm(sales ~ radio)
advertising_fit_newspaper <- lm(sales ~ newspaper)
```
###### OUTPUT 4:
![output4](/img/posts/output4.png "output4")
###### OUTPUT 5:
![output5](/img/posts/output5.png "output5")
###### OUTPUT 6:
![output6](/img/posts/output6.png "output6")

#### Question 5: How accurately can we predict future sales?
The response can be predicted using the following formula (this is not our final model and that's why we will ignore newspaper variable as we have seen that it's not important)
```
sales = 2.938889 + 0.045765 x TV + 0.188530 x radio + e
```

The accuracy associated with this estimate depends on whether we wish to predict an
***individual response*** (for a particular city out of our 200 cities), or the ***average response*** (over a large number of cities). If the former, we use a ***prediction interval***, and if the latter, we use a ***confidence interval***. Prediction intervals will always be wider than confidence intervals because they account for the uncertainty
associated with e, the irreducible error.

