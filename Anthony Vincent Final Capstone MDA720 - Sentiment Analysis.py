#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords


# In[2]:


data = pd.read_csv('/Users/anthonyvincent/Downloads/Apple-Twitter-Sentiment-DFE.csv', encoding = 'iso-8859-1')
data.head()


# In[3]:


data.info()


# In[4]:


data.loc[data['sentiment'] == 'not_relevant', 'sentiment'] = -1
data['sentiment'] = data['sentiment'].astype(int)


# In[5]:


#To predict the Sentiment as positive(numerical value = 1) or 
#negative(numerical value = 0), we need to change them the values
#to those categories. For that the condition will be like if the
#sentiment value is less than or equal to 3, then it is negative(0) else positive(1). 


# In[6]:


#1,2,3->negative(i.e 0)
data.loc[data['sentiment']<=3,'sentiment'] = 0
 
#4,5->positive(i.e 1)
data.loc[data['sentiment']>3,'sentiment'] = 1


# In[7]:


stp_words=stopwords.words('english')
def clean_review(review):
  cleanreview=" ".join(word for word in review.
                       split() if word not in stp_words)
  return cleanreview
 
data['text']=data['text'].apply(clean_review)


# In[8]:


data.head()


# In[9]:


data[['sentiment','text']]
#creation of a new datafram that extracts both the sentiment and text columns that are called


# In[10]:


data['sentiment'].value_counts() #counts the number of sentiment values for negative(the numbner zero) and positive(the number 1)


# In[11]:


consolidated=' '.join(word for word in data['text'][data['sentiment']==0].astype(str))
wordCloud=WordCloud(width=1600,height=800,random_state=21,max_font_size=110)
plt.figure(figsize=(15,10))
plt.imshow(wordCloud.generate(consolidated),interpolation='bilinear')
plt.axis('off')
plt.show()
#displays and generates a word cloud visualization of the data's text 
#calls the text data from the "text" column and where the sentiment label in the data frame is called sentiment


# TF-IDF calculates that how relevant a word in a series or corpus is to a text. The meaning increases proportionally to the number of times in the text a word appears but is compensated by the word frequency in the corpus (data-set). We will be implementing this with the code below.

# In[13]:


cv = TfidfVectorizer(max_features=2500)
X = cv.fit_transform(data['text'] ).toarray()


# In[14]:


from sklearn.model_selection import train_test_split
x_train ,x_test,y_train,y_test=train_test_split(X,data['sentiment'],
                                                test_size=0.25 ,
                                                random_state=42)


# In[15]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
model=LogisticRegression()
 
#Model fitting
model.fit(x_train,y_train)
 
#testing the model
pred=model.predict(x_test)
 
#model accuracy
print(accuracy_score(y_test,pred))


# In[16]:


from sklearn import metrics
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,pred)
 
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm,
                                            display_labels = [False, True])
 
cm_display.plot()
plt.show()


# In[17]:


#Plot a cloud of Negative Tweets


# In[18]:


data_neg = data['text'][:800000]
plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 1000 , width = 1600 , height = 800,
               collocations=False).generate(" ".join(data_neg))
plt.imshow(wc)


# In[19]:


#Plot a cloud of Positive Tweets


# In[20]:


data_pos = data['text'][:800000]
wc = WordCloud(max_words = 1000 , width = 1600 , height = 800,
              collocations=False).generate(" ".join(data_pos))
plt.figure(figsize = (20,20))
plt.imshow(wc)

