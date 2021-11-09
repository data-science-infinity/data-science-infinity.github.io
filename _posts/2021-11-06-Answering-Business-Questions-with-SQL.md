---
layout: post
title: Answering Business Questions with SQL
image: "/posts/john_schnobrich_unsplash.jpg"
tags: [SQL, Data Analysis]
---

We will be answering important business quiestions using SQL. This is very easy and quick. The data we are using is from a supermaket

---

#### Question 1: What were the total sales for each ***product area name*** for July 2020. Return these in the order of highest sales to lowest sales
###### SQL CODE:

```sql
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
```

###### RESULT 1:
![sql1](/img/posts/sql1.png "sql1")

#### Question 2: Return a list of customers who spent more than $500 and had 5 or more ***unique transactions*** in the month of August 2020

```sql
select
 customer_id,
 sum(sales_cost) as total_sales,
 count(distinct(transaction_id)) as total_trans
 
from
 grocery_db.transactions
 
where 
 transaction_date between '2020-08-01' and '2020-08-31'
 
group by
 customer_id
 
having
 sum(sales_cost) > 500 and
 count(distinct(transaction_id)) >= 5;
```
###### RESULT 2:
![sql2](/img/posts/sql2.png "sql2")

#### Question 3: Return data showing, for each ***product area name*** - the ***total sales***, and the percentage of ***overall sales*** that each product area makes up

```sql
with sales as (
select
 b.product_area_name,
 sum(a.sales_cost) as total_sales
 
from
 grocery_db.transactions a
 inner join grocery_db.product_areas b on a.product_area_id=b.product_area_id
 
group by
 b.product_area_name
 
)

select
 product_area_name,
 total_sales,
 total_sales / (select sum(total_sales) from sales) as total_sales_pc
 
from 
 sales;
```
###### RESULT 3:
![sql3](/img/posts/sql3.png "sql3")

