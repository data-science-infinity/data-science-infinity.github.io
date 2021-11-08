---
layout: post
title: Experimental Design in R
image: "/posts/chuttersnap_unsplash.jpg"
tags: [R,Experimental Design, ANOVA]
---
An agricultural scientist wishes to compare the effect of weed control on wheat yield using a herbicide spray regime versus a regime where no spray is used. Analyse the date and report on your conclusions. In your analysis you should complete the following steps and any others that you consider to be appropriate.

---
Check for variance heterogeneity and non-normality and, if detected, take
appropriate action.
To check our residuals to see that our assumptions on variance homogeneity and normality hold we use R and conduct analysis of variance
and look at the plots:

```r
weedcontrol<- read.table("weedcontrol.txt", header=TRUE)
weedcontrol
summary(weedcontrol)
```

###### OUTPUT 1:
![exp](/img/posts/exp.png "exp")

```r
aov1 <- aov(yield ~ fblock + fherbicide * fnitrogen * fpotassium,
data=weedcontrol)

summary(aov1)
```
###### OUTPUT 2:
![exp1](/img/posts/exp1.png "exp1")

```r
par(mfrow=c(2,2))
plot(aov1)
```
###### OUTPUT 3:
![exp2](/img/posts/exp2.png "exp2")

We need to look at the left two plots(aov1) to check if variance is constant.We can see some suggestion that our assumptions don’t hold. Looking at our top left residual plot and checking we can see the fan shape
which means that as mean increases, the variance increases. On our
bottom left standardized residuals plot we can see that the line is not
2
horizontal. This means that our variance is not constant.
On all four of our plots we can see outliers 52, 84, 19 and looking at our
top right QQ plot, we can see that there is something wrong in the tails,
suggesting that probably our error  doesn’t have a normal distribution.
To overcome these problems we need to use a log-transform, conduct analysis of variance again and look at the plots. Later we will back transform
some values of interest using exp function in R.

```r
weedcontrol$y<-log(weedcontrol$yield)
aov2 <- aov(y ~ fblock + fherbicide * fnitrogen * fpotassium,
data=weedcontrol)

summary(aov2)

plot(aov2)
```

###### OUTPUT 4:
![exp3](/img/posts/exp3.png "exp3")

Looking at our new plots(aov2) we can see that the actions we took overcome the problem. On our top left residual plot the fan shape changed,
3
on the left bottom standardized residual plot the red line is closer to horizontal line and our new top right QQ plot shows that the problem with
tails is solved.

Construct an ANOVA table and test for treatment effects.

###### OUTPUT 5:
![exp4](/img/posts/exp4.png "exp4")
