# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 18:36:20 2021

@author: metzm
"""
#verison 1
n = 20
number_range= set(range(2, n+1))
primes_list = []


    prime = number_range.pop()
    primes_list.append(prime)
    multiples =set(range(prime *2, n+1, prime))
    number_range.difference_update(multiples)


#version 2
#upper bound
n = 1000

#number range to be checked 
number_range= set(range(2, n+1))

#empty list to append discovered primes
primes_list = []

#while loop
while number_range:
    prime = number_range.pop()
    primes_list.append(prime)
    multiples =set(range(prime *2, n+1, prime))
    number_range.difference_update(multiples)
  
# print our list of primes
print(primes_list)

#number of primes that were found
prime_count= len(primes_list)

#largest prime
largest_prime = max(primes_list)

# summary
print(f"There are {prime_count} prime numbers between 2 and {n}. The largest of which is {largest_prime}.")


#verison 3

def primes_finder (n):
    #number range to be checked 
    number_range= set(range(2, n+1))
    
    #empty list to append discovered primes
    primes_list = []
    
    #while loop
    while number_range:
        prime = number_range.pop()
        primes_list.append(prime)
        multiples =set(range(prime *2, n+1, prime))
        number_range.difference_update(multiples)
      
    # print our list of primes
    print(primes_list)
    
    #number of primes that were found
    prime_count= len(primes_list)
    
    #largest prime
    largest_prime = max(primes_list)
    
    # summary
    print(f"There are {prime_count} prime numbers between 2 and {n}. The largest of which is {largest_prime}.")
    return primes_list

primes_finder(100)

primes_list = primes_finder(10000)




