---
layout: post
title: Answering Business Questions with SQL
image: "/posts/john_schnobrich_unsplash.jpg"
tags: [SQL, Data Analysis]
---

We will be answering important business quiestions using SQL. This is very easy and quick. The data we are using is from a supermaket

---

#### Question 1: What were the total sales for each productareaname for July 2020.Return these in the order ofhighest sales to lowest sales5) Return a list of all customer_id's that do NOT havea loyalty score (i.e. they are in the customer_detailstable, but not in the loyalty_scores table)

select
 b.product_area_name,
 sum(a.sales_cost) as total_sales
 
from 
 grocery_db.transactions a
 inner join grocery_db.product_areas b on a.product_area_id=b.product_area_id
 
where
 a.transaction_date between '2020-07-01' and '2020-07-31'
 
group by
 b.product_area_name
 
order by
  total_sales desc;



#### Question 2: eturn a list of customers who spent more than$500 and had 5 or more unique transactions in themonth of August 20204) Return a list of duplicate credit scores that existin the customer_details table5) Return the customer_id(s) for the customer(s) whohas/have the 2nd highest credit score. Make sureyour code would work for the Nth highest creditscore as well


#### Question 3: Return data showing, for each product_area_name - thetotal sales, and the percentage of overall sales that eachproduct area makes up
