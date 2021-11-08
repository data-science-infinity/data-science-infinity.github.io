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
weedcontrol$fblock <- factor(weedcontrol$block)
weedcontrol$fherbicide <- factor(weedcontrol$herbicide)
weedcontrol$fnitrogen <- factor(weedcontrol$nitrogen)
weedcontrol$fpotassium <- factor(weedcontrol$potassium)

aov1 <- aov(yield ~ fblock + fherbicide * fnitrogen * fpotassium,
data=weedcontrol)

summary(aov1)

OUTPUT:

Df Sum Sq Mean Sq F value Pr(>F)
fblock 2 0.05 0.03 0.115 0.8912
fherbicide 1 133.73 133.73 604.233 < 2e-16 ***
fnitrogen 3 53.06 17.69 79.913 < 2e-16 ***
fpotassium 4 276.75 69.19 312.602 < 2e-16 ***
fherbicide:fnitrogen 3 61.75 20.58 92.999 < 2e-16 ***
fherbicide:fpotassium 4 17.30 4.33 19.546 3.56e-11 ***
fnitrogen:fpotassium 12 6.94 0.58 2.612 0.0056 **
fherbicide:fnitrogen:fpotassium 12 12.27 1.02 4.620 1.30e-05 ***
Residuals 78 17.26 0.22
---
Signif. codes: 0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
```
