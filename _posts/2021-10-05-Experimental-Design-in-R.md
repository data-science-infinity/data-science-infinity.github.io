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
Code and output in RStudio:

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

