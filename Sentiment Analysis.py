#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


# In[2]:


df = pd.read_csv('F:\mahmoud ali\oasis project\Task 3\\apps.csv')
print(df.head())


# In[3]:


print(df.info())


# In[4]:


print(df.columns)


# In[5]:


print(df['Reviews'].value_counts())


# In[6]:


df.dropna(inplace=True)


# In[7]:


def get_sentiment(rating):
    if rating >= 4.0:
        return 'positive'
    elif rating < 2.5:
        return 'negative'
    else:
        return 'neutral'

df['sentiment'] = df['Rating'].apply(get_sentiment)


# In[8]:


print(df['sentiment'].value_counts())


# In[9]:


import re
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'\W', ' ', str(text))  
    text = text.lower()  
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)  
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)  
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    text = ' '.join([stemmer.stem(word) for word in text.split() if word not in stop_words])
    return text

df['cleaned_reviews'] = df['Reviews'].apply(preprocess_text)


# In[10]:


tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['cleaned_reviews']).toarray()
y = df['sentiment']


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[12]:


model = MultinomialNB()
model.fit(X_train, y_train)


# In[13]:


y_pred = model.predict(X_test)


# In[14]:


print(classification_report(y_test, y_pred, zero_division=1))


# In[15]:


conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[16]:


sns.countplot(df['sentiment'])
plt.title('Sentiment Distribution')
plt.show()


# In[17]:


plt.figure(figsize=(10, 6))
sns.boxplot(x=df['Rating']) 
plt.title('Boxplot of Ratings')
plt.show()


# Recommendations
# Enhance Review Quality: Encourage users to provide detailed and clear reviews to improve the accuracy of sentiment analysis.
# Try More Complex Machine Learning Models: If the current model accuracy is low, consider using advanced models like SVM or neural networks for better results.
# Expanded Sentiment Analysis: Add a more detailed sentiment analysis to differentiate between various impacts of reviews and ratings.
# Customer Engagement Based on Sentiments: Quickly address feedback from customers with negative sentiments to improve their experiences

# In[ ]:





# In[ ]:





# In[ ]:




