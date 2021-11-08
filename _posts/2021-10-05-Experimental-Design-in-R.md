---
layout: post
title: Experimental Design in R
image: "/posts/chuttersnap_unsplash.jpg"
tags: [R,Experimental Design, ANOVA]
---
We will be modeling data om average number of goals for English Premier League and forecasting for the future years

---
Check for variance heterogeneity and non-normality and, if detected, take
appropriate action.
To check our residuals to see that our assumptions on variance homogeneity and normality hold we use R and conduct analysis of variance
and look at the plots:
Code and output in RStudio:
weedcontrol<- read.table("weedcontrol.txt", header=TRUE)
weedcontrol
summary(weedcontrol)
weedcontrol$fblock <- factor(weedcontrol$block)
weedcontrol$fherbicide <- factor(weedcontrol$herbicide)
weedcontrol$fnitrogen <- factor(weedcontrol$nitrogen)
weedcontrol$fpotassium <- factor(weedcontrol$potassium)
aov1 <- aov(yield ~ fblock + fherbicide * fnitrogen * fpotassium,
data=weedcontrol)
1
