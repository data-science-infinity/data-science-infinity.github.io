---
layout: post
title: Tester3
image: "/posts/primes_image.jpeg"
tags: [Python, Primes]
---

In this post I'm going to run through a function in Python that can quickly find all the Prime numbers below a given value.  For example, if I passed the function a value of 100, it would find all the prime numbers below 100!

If you're not sure what a Prime number is, it is a number that can only be divided wholly by itself and one so 7 is a prime number as no other numbers apart from 7 or 1 divide cleanly into it 8 is not a prime number as while eight and one divide into it, so do 2 and 4

Let's get into it!

---
The smallest true Prime number is 2, so we want to start by creating a list of numbers than need checking so every integer between 2 and what we set above as the upper bound which in this case was 20. We use n+1 as the range logic is not inclusive of the upper limit we set there

Instead of using a list, we're going to use a set.  The reason for this is that sets have some special functions that will allow us to eliminate non-primes during our search.  You'll see what I mean soon...

```ruby
number_range = set(range(2, n+1))
```

Let's also create a place where we can store any primes we discover.  A list will be perfect for this job
