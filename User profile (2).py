
# coding: utf-8

# In[12]:


import pandas
import csv
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
user = pandas.read_csv('user.csv',index_col='User_ID')
print(user)


# In[13]:


#user['Age']=user['Age'].astype(int)
user.dtypes


# In[14]:


user.loc['2'] = ['Kevin','M',20,3,'Sci-fi','J.R. Tolkein',5] 
#add a new user in the next line


# In[15]:


print(user.Reading_Habit)


# In[16]:


user.drop(['2'])


# In[17]:


user.loc['2'] = ['Kevin','M',20,3,'Sci-fi','J.R. Tolkein',5] 
user.loc['3'] = ['Shefa','F',16,1,'Fantasy','Suzanne Collinds',5] 


# In[18]:


print(user)


# In[19]:


user.to_csv('user_db.csv')
user_db=pandas.read_csv('user_db.csv',index_col='Name')
#create a new file with the new data


# In[20]:


print(user_db)

