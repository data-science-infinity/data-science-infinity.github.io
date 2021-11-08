---
layout: post
title: Predicting Sales Using Advertising Data in R
image: "/posts/slidebean_unsplash.jpg"
tags: [R, Multiple Linear Regression]
---
How can we predict how much increase in sales will we get after spending money on different types of ads? I am using Advertising data here to show this and any company can collect this kind of data maybe from different branches and answer the question: For how much are we going to increase sales spending the curtain amount on ads or how much money do the company needs to spend on ads to increase sales by X amount of dollars? 

---

Let's import the Advertising data and take a look. On the output below we can see that we have 3 predictors TV, radio, newspaper. The corresponding columns show the amount of money spent on ads for each advertising type. The response variable is sales. All values are in $1000, for example, the values 10.4 in the second row of the sales' column is equivalent to $10,400 sales. The sample size is 200, it's the data collected from two hundred cities.
```R
Advertising <- read.csv("Advertising.csv", head=TRUE)
head(Advertising)
```
###### OUTPUT:
![output](/img/posts/output.png "output")

#### We will be answering 7 questions
#### Question 1: Is there a relationship between advertising and sales?

To answer this question, we will then fit the multiple regression model to test the Ho hypothese. We use ***F-statistic*** for this, there is a relationship between the response and predictors when F-statistic is greater than 1. But how big is big? When the p-value is less than 0.05, we can conclude that there is a relationship. As we can see on the output1 below the p-value(2.2e-16=0.00000000000000022) corresponding to the F-statistic(570.3) is very close to zero, indicating clear evidence of a relationship between advertising and sales.
```r
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

We will examine this further to see if ***collinearity*** be the reason that the confidence interval associated with newspaper is so wide? As a ***rule of thumb***, a VIF value that is bigger than 5 or 10 indicates a problematic of collinearity. From the output 3 we can see that the VIF scores are 1.005, 1.145, and 1.145 for TV, radio, and newspaper, suggesting no evidence of collinearity.
```R
library(car)
vif(advertising_fit)
```
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
***individual response*** (for a particular city out of our 200 cities), or the ***average response*** (over a large number of cities). If the former, we use a ***prediction interval***, and if the latter, we use a ***confidence interval***. As we can see ont the output7, prediction intervals are wider than confidence intervals and this will always be the case because they account for the uncertainty associated with e, the irreducible error.

```R
newdata=data.frame(TV = 1, radio = 1, newspaper = 1)
predict(advertising_fit, newdata, interval = 'prediction')
predict(advertising_fit, newdata, interval = 'confidence')
```

###### OUTPUT 7:
![output7](/img/posts/output7.png "output7")

#### Question 6: Is the relationship linear?
The residual plots can be used in order to identify non-linearity. If the relationships are linear, then the residual plots should display no pattern. We will look only at the plot of residulas vs. predicted (of fitted) values. ***U-shape*** we can see on output8 below provides evidance of non-linearity. We can use non-linear transformations of the predictors, such as log X, square root of X, and X^2, in the linear regression model in order to accommodate non-linear relationships. 
```R
par(mfrow=c(2,2))
plot(advertising_fit, lwd=3)
```
###### OUTPUT 8:
![output8](/img/posts/output8.png "output8")

#### Question 7: Is there synergy among the advertising media?
The standard linear regression model assumes an additive relationship between the predictors and the response, meaning that the effect of changes in one of the predictors (let's say money spent on TV ads) on the response (sales) is independent of the values of the other predictors (radio and newspaper). An additive model is easy to interpret. However, the additive assumption may be unrealistic for certain data sets. Output9 suggested that the Advertising data may not be additive. A small p-value (output10) associated with the interaction term indicates the presence of such relationships. Including an interaction term in the model results in a substantial increase in R-squared, from around 90% to almost 97%!

###### OUTPUT 9:
![output9](/img/posts/output9.png "output9")

```R
advertising_fit_int <- lm(sales ~ TV * radio)
summary(advertising_fit_int)
```
###### OUTPUT 10:
![output10](/img/posts/output10.png "output10")

---

We have properly fitted the ***multiple linear regression model*** with the interaction term to the model, we can make predictions using the following formula:


```
sales = 6.750e+00 + 1.910e-02 x TV + 2.886e-02 x radio +  1.086e-03 x TV x radio + e
```


