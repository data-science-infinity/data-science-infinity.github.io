---
layout: post
title: Experimental Design in R
image: "/posts/chuttersnap_unsplash.jpg"
tags: [R,Experimental Design, ANOVA]
---
An agricultural scientist wishes to compare the effect of weed control on wheat yield using a herbicide spray regime versus a regime where no spray is used. Analyse the date and report on your conclusions. In your analysis you should complete the following steps and any others that you consider to be appropriate.

---
This project was the requirement for my module on the Experimental Design for my MSc Statistical Data Science course at the University of Kent. The aim of the project was to analyse the outcomes of the experiment held by an agricultural scientist who wanted to compare the effect of weed control on wheat yield using a ***herbicide spray regime*** versus a regime where no spray is used. He also wished to see how this interacts with different fertiliser regimes, consisting of all combinations of 4 ***nitrogen levels***, and 5 ***potassium levels***. The experiment thus has 40 treatment combinations which are applied to 120 plots laid out in ***three blocks*** of 40 plots per block. At the end of the season the wheat is harvested and the yield recorded for each plot. One of the tasks on this assessment was to report analysis and include a summary paragraph detailing conclusions for the ***nontechnical reader***. I

The resulting data was a table containing mean values from these 120 plots. I used ANOVA to analyse data. In order to be able to use this method, we need to check for variance heterogeneity and non-normality and, if detected, take appropriate action. To check our residuals to see that our assumptions on variance homogeneity and normality hold we use R and conduct analysis of variance and look at the plots:

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

Looking at our top left residual plot and checking we can see the ***fan*** shape which means that as mean increases, the variance increases. On our
bottom left standardized residuals plot we can see that the line is not horizontal. This means that our variance is not constant. On all four of our plots we can see outliers 52, 84, 19 and looking at our top right QQ plot, we can see that there is something wrong in the tails, suggesting that probably our error doesn’t have a normal distribution.
To overcome these problems we need to use a log-transform, conduct analysis of variance again and look at the plots. Later we will back transform some values of interest using exp function in R.

```r
weedcontrol$y<-log(weedcontrol$yield)
aov2 <- aov(y ~ fblock + fherbicide * fnitrogen * fpotassium,
data=weedcontrol)

plot(aov2)
```

###### OUTPUT 4:
![exp3](/img/posts/exp3.png "exp3")

Looking at our new plots(aov2) we can see that the actions we took overcome the problem. On our top left residual plot the fan shape changed, on the left bottom standardized residual plot the red line is closer to horizontal line and our new top right QQ plot shows that the problem with tails is solved.

We can now move to conducting an ANOVA table and testing for treatment effects.

###### OUTPUT 5:
![exp4](/img/posts/exp4.png "exp4")

```r
aov2 <- aov(y ~ fblock + fherbicide * fnitrogen * fpotassium,
data=weedcontrol)

summary(aov2)
```


###### OUTPUT 6:
![exp5](/img/posts/exp5.png "exp5")

I analysed the outcome using the ***hierarcical principle***. First looked at the second order interaction to see if it’s significant, in our case it’s not, then we move up
to the first order interactions, in our case only ***herbicide with nitrogen interaction*** is significant, which means that we will look further only at potassium’s main effect. It turns out that potassium’s main effect is significant.

Next, we need to compare treatments to recommend an ***optimum treatment*** combination. We will compare means and interpret the results

```r
library(emmeans)
emmeans(aov2, pairwise ~ fherbicide:fnitrogen)
emmeans(aov2, pairwise ~ fpotassium)
$emmeans
```

###### OUTPUT 7:
![exp6](/img/posts/exp6.png "exp6")

```r
$contrasts
```

###### OUTPUT 8:
![exp7](/img/posts/exp7.png "exp7")

![exp7a](/img/posts/exp7a.png "exp7a")

```r
emmeans(aov2, pairwise ~ fpotassium)
# NOTE: Results may be misleading due to involvement in interactions
$emmeans
```

###### OUTPUT 9:
![exp8](/img/posts/exp8.png "exp8")

```r
$contrasts
```
###### OUTPUT 10:
![exp9](/img/posts/exp9.png "exp9")


+ text
