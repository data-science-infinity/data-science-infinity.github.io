---
layout: post
title: Earthquake Tracker - Tableau Dashboard
image: "/posts/quake_dash.jpg"
tags: [Tableau, Data Viz]
---

This interactive Tableau dashboard tracks global earthquake activity for a given data set of a 30-day period.

***NOTE:** Due to formatting troubles between Tableau, Ruby, and git-pages, it is recommended to view the dashboard on Tableau Public itself by clicking "View on Tableau Public" on the bottom left of the dashboard below.*

* Take in the overall data summary for earthquakes over a 30-day period. 
* See the frequency of earthquakes in each location as well as the average and maximum magnitude
* For finer detail, scroll for a summary of particular days in the upper right hand slide bar
* Hover your mouse over the data points on the map to easily see earthquake information, such as locations and magnitudes for each recorded event

___

<!-- Wrapper that allows horizontal scrolling on narrower GitHub Pages containers -->
<div class="tableauPlaceholder" id="viz1785267581845" style="position: relative;">
  <noscript>
    <a href="#">
      <img 
        alt="Earthquake Tracker" 
        src="https://public.tableau.com/static/images/Ea/EarthquakeDashboard_17852091836780/EarthquakeTracker/1_rss.png" 
        style="border: none;" 
      />
    </a>
  </noscript>
  <object class="tableauViz" style="display: none;">
    <param name="host_url" value="https://public.tableau.com/" />
    <param name="embed_code_version" value="3" />
    <param name="site_root" value="" />
    <param name="name" value="EarthquakeDashboard_17852091836780/EarthquakeTracker" />
    <param name="tabs" value="yes" />
    <param name="toolbar" value="yes" />
    <param name="static_image" value="https://public.tableau.com/static/images/Ea/EarthquakeDashboard_17852091836780/EarthquakeTracker/1.png" />
    <param name="animate_transition" value="yes" />
    <param name="display_static_image" value="yes" />
    <param name="display_spinner" value="yes" />
    <param name="display_overlay" value="yes" />
    <param name="display_count" value="yes" />
    <param name="language" value="en-US" />
  </object>
</div>

<script type="text/javascript">
  var divElement = document.getElementById('viz1785267581845');
  var vizElement = divElement.getElementsByTagName('object')[0];

  if (divElement.offsetWidth > 800) {
    vizElement.style.width = '100%';
    vizElement.style.height = (divElement.offsetWidth * 0.75) + 'px';
  } else if (divElement.offsetWidth > 500) {
    vizElement.style.width = '100%';
    vizElement.style.height = (divElement.offsetWidth * 0.75) + 'px';
  } else {
    vizElement.style.width = '100%';
    vizElement.style.minHeight = '1450px';
    vizElement.style.maxHeight = (divElement.offsetWidth * 1.77) + 'px';
  }

  var scriptElement = document.createElement('script');
  scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
  vizElement.parentNode.insertBefore(scriptElement, vizElement);
</script>
___

# **Data Disclaimer**: 
This earthquake data in this dashboard is not realistic whatsoever. The frequency of high magnitude quakes are drastically exaggerated to provide diverse datapoints to demonstrate data visualization in Tableau. If we were to receive legitimate data from an agency like the USGS, we'd be able to simply upload the real world data into our proof-of-concept dashboard

### Fun Fact(s): 
* The greatest magnitude of an earthquake ever recorded was in Chili in 1960 at a magnitude of 9.5. 
* Earthquakes cannot exceed a magnitude of 10 because no tectonic fault line is long enough to produce one. A magnitude 10 earthquake would require a fault wrapping all the way around the equator
* The USGS (United States Geological Survey) estimates an average of 16 major earthquakes per year, with a magnitude of 7.0 or higher 
