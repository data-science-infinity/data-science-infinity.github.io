---
layout: post
title: My Sample Project
image: "/posts/coffee_python.jpg"
tags: [Python, Primes]
---

In this post I'm going to run through a function in Python that can quickly find all the Prime numbers below a given value.  For example, if I passed the function a value of 100, it would find all the prime numbers below 100!

If you're not sure what a Prime number is, it is a number that can only be divided wholly by itself and one so 7 is a prime number as no other numbers apart from 7 or 1 divide cleanly into it 8 is not a prime number as while eight and one divide into it, so do 2 and 4

Let's get into it!

---

First let's start by setting up a variable that will act as the upper limit of numbers we want to search through. We'll start with 20, so we're essentially wanting to find all prime numbers that exist that are equal to or smaller than 20

```ruby
while number_range:
    prime = number_range.pop()
    primes_list.append(prime)
    multiples = set(range(prime*2, n+1, prime))
    number_range.difference_update(multiples)
```

```python
while number_range:
    prime = number_range.pop()
    primes_list.append(prime)
    multiples = set(range(prime*2, n+1, prime))
    number_range.difference_update(multiples)
```

![Philadelphia's Magic Gardens. This place was so cool!](/img/posts/coffee_python.jpg "Philadelphia's Magic Gardens")
