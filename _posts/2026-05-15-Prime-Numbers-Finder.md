---
layout: post
title: Prime Numbers Finder with Python
image: "/posts/prime_nums.jpg"
tags: [Python, Primes]
---

Hello! In this post, I'll be walking through a Python function I wrote that returns a list of all prime numbers that are below a specified input number, **n**. For example, if we passed in a value n=10, this function would return a list of all prime numbers below the specified number 10!

For reference, a prime number is an integer that can only be divided wholly by itself and 1. In our example of n=10, we would expect an output as follows:
```python
>>> {2, 3, 5, 7}
```

--- 
Before we jump into the final script, we'll be walking through each functional step of the code. In the end, we'll have a while loop that loops through a range of numbers up to the defined number, **n** that is able to return a list of prime numbers within that range. 

First, let's set our input value **n**, the number in which we want to find all prime numbers up to.

```python
n = 20
```

Next, let's define the range of values, up to n = 20.

The lowest prime number to exist is 2. Because of this, we want our initial list of possible prime numbers, **number_range**, to be a list of numbers starting from 2 and ending at specified input "n". 

Instead of an ordinary list, however, we'll be using a set for our number_range. Sets are a good choice as sets are faster than lists and contain unique values, however, Stay tuned to find out why!

```python
number_range = set(range(2, n+1))
>>> {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}
```
The end goal is to loop through this number_range and extract each value from this set with the *pop* method to determine whether each value is a prime number. If the routine determines a value is prime, we'll append it onto the new **primes_list** list. Let's initialize that empty list!

```python
primes_list = []
```

The *pop* method removes an element from a list or set, and returns that value to us. 2 is the first value in our number_range, and is the lowest prime number to exist. Let's pop it out and assign it to **prime**

```python
prime = number_range.pop()
print(prime)
>>> 2
print(number_range)
>>> {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}
```

After the pop is applied, value 2 is no longer in our number_range set. We'll then add this first prime number **prime** to our empty **primes_list**

```python
primes_list.append(prime)
print(primes_list)
>>> [2]
```

All multiples of 2 are by default not prime numbers, since they can be divided by 2 in addition to itself and 1. Let's define a new set for all multiples of 2 that are lower than the defined n=20 value.

```python
multiples = set(range(prime*2, n+1, prime))
```

The range was determined as follows: range(start, stop, step) 

For the START in our range, we can start with prime*2, which will be the first multiple of the prime number.  
For the STOP in our range, we want to stop at the index of our number n=20, which is the value of n+1.  
For the STEP in our range, we want multiples of our prime, so we want to increment in steps of this popped out prime number so we can put in our **prime** list.

Now we can see that this range worked in our output as we have all multiples of 2 up to our n value of 20!

```python
print(multiples)
>>> {4, 6, 8, 10, 12, 14, 16, 18, 20}
```

Here's where the magic happens! Why did we use sets? With sets, we can use a special functionality called **difference_update**. 

Before we apply the **difference_update** method, let's look at our two sets again.

```python
print(number_range)
>>> {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}

print(multiples)
>>> {4, 6, 8, 10, 12, 14, 16, 18, 20}
```

**difference_update** will be used to update our number_range by dropping the values that exist in the **multiples** list we defined.

```python
number_range.difference_update(multiples)
print(number_range)
>>> {3, 5, 7, 9, 11, 13, 15, 17, 19}
```

Our number_range list is now trimmed down to not include multiples of 2, as we've determined the **multiples** values are not prime. This vastly reduces how many numbers in the number_range we have left to assess. The same steps above can now be applied to the the first number in the new number_range set, 3. We would pop 3 into the primes_list, find all multiples of 3 up to value **n=20**, and then difference_update to remove the non-prime multiples from our number_range set.

We now have all the pieces to make this loop work, where we can define **n** and have the script return the **primes_list** on it's own! 

Let's assemble our while loop and up the ante, setting our **n** value to 1000:

```python
n = 1000

# define number range
number_range = set(range(2, n+1))

# create empty list to append prime numbers to
primes_list = []

# iterate until the number_range set is empty
while number_range:
    prime = number_range.pop()
    primes_list.append(prime)
    multiples = set(range(prime*2, n+1, prime))
    number_range.difference_update(multiples)
	
print(primes_list)
>>> [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997]
```

Success! Now let's throw this all into a function.

A list of the prime numbers was our goal, however, we can also return a few more details to the user about the extracted primes_list. We can print the amount of prime numbers found in our range as well as what the highest prime number is in the set.

```python
def primes_finder(n):
    
    # define number range
    number_range = set(range(2, n+1))

    # create empty list to append prime numbers to
    primes_list = []

    # iterate until the number_range set is empty
    while number_range:
        prime = number_range.pop()
        primes_list.append(prime)
        multiples = set(range(prime*2, n+1, prime))
        number_range.difference_update(multiples)
    
	# count how many primes in primes_list output
    prime_count = len(primes_list)
	# find the largest prime number
    largest_prime = max(primes_list)
	# print the details
	print(f"There are {prime_count} prime numbers between 1 and {n}, the largest of which is {largest_prime}")
	
	>>> There are 168 prime numbers between 1 and 1000, the largest of which is 997
```

With our function now defined, it's as easy as running **primes_finder(n)** where n can be any numerical value of choice.

```python
primes_finder(1000000)
>>> There are 78498 prime numbers between 1 and 1000000, the largest of which is 999983
```

And there we have it! Feel free to copy the function and try it for yourself! :)

**A NOTE ON POP():**

In this script, we demonstrated the **pop** functionality, however, using pop() on a set could be risky in some instances. 

Pop() extracts the lowest value of a list or set, which seems confusing since sets are unordered by definition. Under the Python covers, however, the elements in the set are internally stored with an order determined by the hash code of each element. Occassionally, this hash may provide a value that is not the lowest.

To safeguard against this, we can replace the definition of of **prime** value to from *prime = number_range.pop()* to..

```python
prime = min(sorted(number_range))
number_range.remove(prime)
```

This new definition of prime ascertains that we are always pulling the lowest value out of our number_range. This method, however, is slower as we are sorting the number_range set each time we define **prime**.
