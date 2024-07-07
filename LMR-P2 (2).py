#!/usr/bin/env python
# coding: utf-8

# Python assignemnt of linear regression model 

# In[1]:


from matplotlib import pyplot as plt


# In[2]:


import pandas as pd


# In[3]:


import numpy as np


# In[5]:


import sklearn


# In[6]:


from sklearn.linear_model import LinearRegression 


# In[7]:
import sys
import csv

if len (sys.argv) < 2:
   sys.exit(1)


argument = sys.argv[1]
t=[]
x1 =[]
y1=[]

with open (argument ,'r', newline='') as file:
    data = csv.reader(file)
    next(data)
    for row in data:
        x1.append(row[1])
        y1.append(row[0])
       
    

        


# In[8]:

x2 = np.array(x1, dtype= float)
y2= np.array(y1, dtype= float)


# In[9]:







# In[10]:


plt.scatter(x2, y2)
plt.show()
plt.savefig('py_orig.png')

# In[11]:


x= np.array(x2).reshape((-1,1))
y= np.array(y2)
model = LinearRegression ()
model.fit (x, y)


# In[12]:


pred= model.predict(x)


# In[13]:


plt.plot(x2, pred, label = 'Linear Regression Model', color = "red")
plt.scatter(x2, y2, label = 'Scatter Plot', color = "blue")
plt.title("Linear Regression Model Assignment - Python")
plt.xlabel("X data")
plt.ylabel("Y data")
plt.legend()
plt.show()

plt.savefig('py_lm.png')
# In[ ]:




