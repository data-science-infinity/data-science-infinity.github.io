---
layout: post
title: Tester1
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

```ruby
primes_list = []
```

We're going to end up using a while loop to iterate through our list and check for primes, but before we construct that I always it valuable to code up the logic and iterate manually first.  This means I can check that it is working correctly before I set it off to run through everything on it's own

So, we have our set of numbers (called number_range to check all integers between 2 and 20. Let's extract the first number from that set that we want to check as to whether it's a prime. When we check the value we're going to check if it is a prime...if it is, we're going to add it to our list called primes_list...if it isn't a prime we don't want to keep it
