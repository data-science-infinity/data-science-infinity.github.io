---
layout: post
title: Finding Prime Numbers with Python
image: "/posts/primes_image.jpeg"
tags: [Python, Primes]
---

In this post I'm going to run through a function in Python that can quickly find all the Prime numbers below a given value.  For example, if I passed the function a value of 100, it would find all the prime numbers below 100!

If you're not sure what a Prime number is, it is a number that can only be divided wholly by itself and one so 7 is a prime number as no other numbers apart from 7 or 1 divide cleanly into it 8 is not a prime number as while eight and one divide into it, so do 2 and 4

Let's get into it!

---

First let's start by setting up a variable that will act as the upper limit of numbers we want to search through. We'll start with 20, so we're essentially wanting to find all prime numbers that exist that are equal to or smaller than 20

```ruby
n = 20
```

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

There is a method which will remove an element from a list or set and provide that value to us, and that method is called *pop*

```ruby
print(number_range)
>>> {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}
```
If we use pop, and assign this to the object called **prime** it will *pop* the first element from the set out of **number_range**, and into **prime**

```ruby
prime = number_range.pop()
print(prime)
>>> 2
print(number_range)
>>> {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}
```

Now, we know that the very first value in our range is going to be a prime...as there is nothing smaller than it so therefore nothing else could possible divide evenly into it.  As we know it's a prime, let's add it to our list of primes...

```ruby
primes_list.append(prime)
print(primes_list)
>>> [2]
```

Now we're going to do a special trick to check our remaining number_range for non-primes. For the prime number we just checked (in this first case it was the number 2) we want to generate all the multiples of that up to our upper range (in our case, 20).

We're going to again use a set rather than a list, because it allows us some special functionality that we'll use soon, which is the magic of this approach.

```ruby
multiples = set(range(prime*2, n+1, prime))
```

Remember that when created a range the syntax is range(start, stop, step). For the starting point - we don't need our number as that has already been added as a prime, so let's start our range of multiples at 2 * our number as that is the first multiple, in our case, our number is 2 so the first multiple will be 4. If the number we were checking was 3 then the first multiple would be 6 - and so on.

For the stopping point of our range - we specify that we want our range to go up to 20, so we use n+1 to specify that we want 20 to be included.

Now, the **step** is key here.  We want multiples of our number, so we want to increment in steps *of our* number so we can put in **prime** here

Lets have a look at our list of multiples...

```ruby
print(multiples)
>>> {4, 6, 8, 10, 12, 14, 16, 18, 20}
```

The next part is the magic I spoke about earlier, we're using the special set functionality **difference_update** which removes any values from our number range that are multiples of the number we just checked. The reason we're doing this is because if a number is a multiple of anything other than 1 or itself then it is **not a prime number** and can remove it from the list to be checked.

Before we apply the **difference_update**, let's look at our two sets.

```ruby
print(number_range)
>>> {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}

print(multiples)
>>> {4, 6, 8, 10, 12, 14, 16, 18, 20}
```

**difference_update** works in a way that will update one set to only include the values that are *different* from those in a second set

To use this, we put our initial set and then apply the difference update with our multiples

```ruby
number_range.difference_update(multiples)
print(number_range)
>>> {3, 5, 7, 9, 11, 13, 15, 17, 19}
```

When we look at our number range now, all values that were also present in the multiples set have been removed as we *know* they were not primes

This is amazing!  We've made a massive reduction to the pool of numbers that need to be tested so this is really efficient. It also means the smallest number in our range *is a prime number* as we know nothing smaller than it divides into it...and this means we can run all that logic again from the top!

Whenever you can run sometime over and over again, a while loop is often a good solution.

Here is the code, with a while loop doing the hard work of updated the number list and extracting primes until the list is empty.

Let's run it for any primes below 1000...

```ruby
n = 1000

# number range to be checked
number_range = set(range(2, n+1))

# empty list to append discovered primes to
primes_list = []

# iterate until list is empty
while number_range:
    prime = number_range.pop()
    primes_list.append(prime)
    multiples = set(range(prime*2, n+1, prime))
    number_range.difference_update(multiples)
```

Let's print the primes_list to have a look at what we found!

```ruby
print(primes_list)
>>> [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997]
```

Let's now get some interesting stats from our list which we can use to summarise our findings, the number of primes that were found, and the largest prime in the list!

```ruby
prime_count = len(primes_list)
largest_prime = max(primes_list)
print(f"There are {prime_count} prime numbers between 1 and {n}, the largest of which is {largest_prime}")
>>> There are 168 prime numbers between 1 and 1000, the largest of which is 997
```

Amazing!

The next thing to do would be to put it into a neat function, which you can see below:

```ruby
def primes_finder(n):
    
    # number range to be checked
    number_range = set(range(2, n+1))

    # empty list to append discovered primes to
    primes_list = []

    # iterate until list is empty
    while number_range:
        prime = number_range.pop()
        primes_list.append(prime)
        multiples = set(range(prime*2, n+1, prime))
        number_range.difference_update(multiples)
        
    prime_count = len(primes_list)
    largest_prime = max(primes_list)
    print(f"There are {prime_count} prime numbers between 1 and {n}, the largest of which is {largest_prime}")
```

Now we can jut pass the function the upper bound of our search and it will do the rest!

Let's go for something large, say a million...

```ruby
primes_finder(1000000)
>>> There are 78498 prime numbers between 1 and 1000000, the largest of which is 999983
```

That is pretty cool!

I hoped you enjoyed learning about Primes, and one way to search for them using Python.

---

###### Important Note: Using pop() on a Set in Python

In the real world - we would need to make a consideration around the pop() method when used on a Set as in some cases it can be a bit inconsistent.

The pop() method will usually extract the lowest element of a Set. Sets however are, by definition, unordered. The items are stored internally with some order, but this internal order is determined by the hash code of the key (which is what allows retrieval to be so fast). 

This hashing method means that we can't 100% rely on it successfully getting the lowest value. In very rare cases, the hash provides a value that is not the lowest.

Even though here, we're just coding up something fun - it is most definitely a useful thing to note when using Sets and pop() in Python in the future!

The simplest solution to force the minimum value to be used is to replace the line...

```ruby
prime = number_range.pop()
```

...with the lines...

```ruby
prime = min(sorted(number_range))
number_range.remove(prime)
```

...where we firstly force the identification of the lowest number in the number_range into our prime variable, and following that we remove it.

However, because we have to sort the list for each iteration of the loop in order to get the minimum value, it's slightly slower than what we saw with pop()!


