#!/usr/bin/env python
# coding: utf-8

# # Prateek Patel
# #### Mathematical modelling and data science Engineer for Hybrid electric Vehicle development
# Contact  : prateekpatel.in@gmail.com <br>

# In[3]:


# Past Experience
import plotly
import plotly.figure_factory as ff
plotly.offline.init_notebook_mode(connected=True)

df = [dict(Task="Maruti Suzuki", Start='2009-01', Finish='2015-10'),
      dict(Task="Affluent [JLR]", Start='2015-10', Finish='2016-04'),
      dict(Task="Altran [Ford]", Start='2016-06', Finish='2020-01'),
      dict(Task="Vantage Power", Start='2020-01', Finish='2021')]


fig = ff.create_gantt(df, title='Past Experience', bar_width=0.3, showgrid_x=True, showgrid_y=True)
plotly.offline.iplot(fig, filename='Past Experience')


# In[26]:


# Skills
skills = [("big data", 40, 80), ("python", 100, 90), ("matlab", 55, 70),
         ("machine-learning", 30, 60), ("stats", 70, 80), ("linux", 50, 40),
         ("data-science", 90, 70), ("math-modelling", 50, 90),
         ("project-management", 20, 30), ("HEVs", 80,60), ('GIT', 60,50)]

from matplotlib import pyplot as plt

def text_size(total):
    return 8 + total / 200 * 20

plt.figure()
for skill, time, confidence in skills:
    from matplotlib import pyplot as plt
    plt.text(time, confidence, skill,
            ha='center', va='center',
            size=text_size(time + confidence))
            
plt.title("Skills", size=14)
plt.xlabel("time spent")
plt.ylabel("confidence")
plt.axis([0, 100, 0, 100])
plt.xticks([])
plt.yticks([])
plt.show()

