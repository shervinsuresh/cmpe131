
# coding: utf-8

# In[389]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
list_len=25;
curr_user=1;


# In[390]:


books = pd.read_csv('books.csv', encoding = "ISO-8859-1")
books.head()


# In[391]:


books.shape


# In[392]:


books.columns


# In[393]:


ratings = pd.read_csv('ratings.csv', encoding = "ISO-8859-1")
#ratings.head()


# In[394]:


book_tags = pd.read_csv('book_tags.csv', encoding = "ISO-8859-1")
#book_tags.head()


# In[395]:


tags = pd.read_csv('tags.csv')
#tags.tail()


# In[396]:


tags_join_DF = pd.merge(book_tags, tags, left_on='tag_id', right_on='tag_id', how='inner')
#tags_join_DF.head()


# In[397]:


to_read = pd.read_csv('to_read.csv')
#to_read.head()


# In[398]:


tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(books['authors'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# In[399]:


cosine_sim


# In[400]:


# Build a 1-dimensional array with book titles
titles = books['title']
indices = pd.Series(books.index, index=books['title'])

# Function that get book recommendations based on the cosine similarity score of book authors
def title_recommendations(authors):
    idx = indices[authors]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:(list_len+1)]
    book_indices = [i[0] for i in sim_scores]
    return titles.iloc[book_indices]


# In[401]:


title_recommendations('The Hobbit').head(list_len)


# In[402]:


books_with_tags = pd.merge(books, tags_join_DF, left_on='book_id', right_on='goodreads_book_id', how='inner')


# In[403]:


tf1 = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix1 = tf1.fit_transform(books_with_tags['tag_name'].head(10000))
cosine_sim1 = linear_kernel(tfidf_matrix1, tfidf_matrix1)


# In[404]:


cosine_sim1


# In[405]:


# Build a 1-dimensional array with book titles
titles1 = books['title']
indices1 = pd.Series(books.index, index=books['title'])

# Function that get book recommendations based on the cosine similarity score of books tags
def tags_recommendations(title):
    idx = indices1[title]
    sim_scores = list(enumerate(cosine_sim1[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]
    book_indices = [i[0] for i in sim_scores]
    return titles.iloc[book_indices]


# In[406]:


temp_df = books_with_tags.groupby('book_id')['tag_name'].apply(' '.join).reset_index()
temp_df.head()


# In[407]:


books = pd.merge(books, temp_df, left_on='book_id', right_on='book_id', how='inner')


# In[408]:


books.head()


# In[409]:


books['corpus'] = (pd.Series(books[['authors', 'tag_name']]
                .fillna('')
                .values.tolist()
                ).str.join(' '))


# In[410]:


tf_corpus = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix_corpus = tf_corpus.fit_transform(books['corpus'])
cosine_sim_corpus = linear_kernel(tfidf_matrix_corpus, tfidf_matrix_corpus)

# Build a 1-dimensional array with book titles
titles = books['title']
indices = pd.Series(books.index, index=books['title'])

# Function that get book recommendations based on the cosine similarity score of books tags
def corpus_recommendations(title):
    idx = indices1[title]
    sim_scores = list(enumerate(cosine_sim_corpus[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]
    book_indices = [i[0] for i in sim_scores]
    return titles.iloc[book_indices]

corpus_recommendations("The Hobbit")


# In[415]:


user= pd.read_csv('user_db.csv', encoding = "ISO-8859-1")
user.dtypes


# In[412]:


print(user)


# In[413]:


curr_data=(user.loc[user['User_ID'] == curr_user])
print(curr_data)


# In[414]:


author = curr_data['Author_Preference'].values[0]
print(author)


# In[375]:


author_fav=(books.loc[books['authors'] == author])


# In[377]:


author_fav[author_fav['ratings_5']==author_fav['ratings_5'].max()]


# In[379]:


user_title = author_fav['title'].values[0]
print(user_title)


# In[382]:


title_recommendations(user_title).head(list_len)

