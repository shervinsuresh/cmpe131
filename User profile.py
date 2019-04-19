
# coding: utf-8

# In[112]:


import pandas
import csv
df = pandas.read_csv('user.csv', index_col='User ID')
print(df)


# In[113]:


df.loc['2'] = ['Kevin','1','1','1'] 
#add a new user in the next line


# In[114]:


print(df)


# In[115]:


df.loc['3']=['Matt','1','1','0']


# In[116]:


print(df)


# In[117]:


df.drop(['2'])
#remove a user by number # kevin was a User ID


# In[118]:


df.loc['2']=['Kevin','1','1','1']


# In[119]:


print(df)


# In[120]:


df.to_csv('user_db.csv')
user_db=pandas.read_csv('user_db.csv',index_col='Name')
#create a new file with the new data


# In[121]:


print(user_db)

