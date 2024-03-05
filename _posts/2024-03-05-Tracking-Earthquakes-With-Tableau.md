---
layout: post
title: Earthquake Dashboard using Tableau
image: "/posts/tableau-map-image.png"

tags: [Tableau, Data Visualization]
---

I've been tasked by an Earthquake monitoring agency to help analyze and visualize global earthquake patterns. I've been handed a 30-day sample of data.
I've outline my process for this project below.

<iframe seamless frameborder="0" src="https://public.tableau.com/views/EarthquakeAnalysis_17083177758890/DSIEarthquakeTracker?:language=en-US&:sid=&:display_count=n&:origin=viz_share_link" width = '1090' height = '900'></iframe>

---

### Situation:

**Create a dashboard to analyze a 30 day sample of data from an Earthquake monitoring agency. The dashboard needs to have the following features according to stakeholders:**

* A map showing where Earthquakes took place, it must show their intensity in terms of magnitude.
* Have a list of the top 10 largest earthquakes in terms of magnitude for easily isolation.
* Have a breakdown of the percentage of earthquakes that occurred in each broad location.
* Then have a look at the each of the countries within each broad location. This should show how many earthquakes took place, what the average magnitude was, and what the maximum magnitude was.
* Finally, all visuals need to be controlled by a single date filter to observe the each of the visuals on a day by day basis.

---

### Task:
I need to develop a Tableau dashboard to visualize earthquake data comprehensively so that my stakeholders will be able to quickly identify meaningful insights or trends related to this 30 day sample.

---
### Action:

1. **Track Largest Earthquakes:** Create a table to view the 10 largest earthquakes on a specific date or all-time for this 30-day sample.
2. **Geographical Distribution:** Create world map to see the frequency of earthquake for a region as well as their magnitude.
3. **Broad Location Percentage:** Observe the percentage of earthquakes categorized broadly - North America, South America, Central America, Asia, Oceania, Europe, the Middle East, and Africa.
4. **Country-wise Frequency:** Analyze the frequency of earthquakes within for the countries within each of the broad locations above, providing insights into regions prone to seismic activity.

#### Magnitude Analysis
4. **Average Magnitude:** Understand the average magnitude of earthquakes for different countries on a given day or all all time.
5. **Maximum Magnitude:** Identify the maximum magnitudes recorded for different countries on a given day or all time.

---
### Result
The created dashboard allows users to:
1. Identify quick insights into earthquakes for a 30 day sample on a given day or for the duration of that 30 days.
2. Have a quick look at seismic activity across the globe by zoning in on areas with high frequency and differentiate by magnitude with efficient use of color.
3. Have access to a country level analysis for frequency, average magnitude, or maximum magnitude.

---

### Example Analysis:

##### What are the most earthquake-prone regions during this time period and on July 25th, 2022.
* All time: North America has the highest percentage of Earthquakes in this dataset at 53% followed by Central America at 20%.
* July 25th, 2022: North America's share of global earthquakes increased to 55%. On this day, Central America and Asia are very close with shares of 12% and 11% respectively.

---

### If given more time I would:
* Explore magnitude trends over time. At what point in this sample were magnitudes at their peak. This could help alert safety authorities to have high awareness during a certain period of the year.
