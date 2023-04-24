#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
from bs4 import BeautifulSoup
import re


# In[2]:


df = pd.read_csv('/Users/anthonyvincent/Downloads/iphone14-query-tweets.csv.zip')


# In[3]:


df.head() #showing the data


# In[4]:


df.info() #display info about pandas data frame


# In[5]:


df.tweet_text #access the column name tweet_text and display info of pandas datafram


# In[6]:


df["clean_tweet_text"] = df["tweet_text"].replace(r'http\S+', '', regex=True).replace(r'www\S+', '', regex=True)
df[["tweet_text","clean_tweet_text"]].iloc[94807]
# creates new column, use replacve to generate new column values, cleans up tweet_text column data


# In[7]:


df.head(5) #show data


# In[8]:


df["clean_tweet_text"].replace(r'\d+', ' ', regex = True, inplace = True)
df[["tweet_text","clean_tweet_text"]] # show cleaned data


# In[9]:


def deEmojify(inputString):
    return inputString.encode('ascii', 'ignore').decode('ascii')
df["clean_tweet_text"] = df["clean_tweet_text"].apply(lambda s: deEmojify(s))
df[['tweet_text','clean_tweet_text']].iloc[12]
#defines function, removes unicode, cleans the tweet_text column, and shows it without the unicodes


# In[11]:


def rem_en(input_txt):
    words = input_txt.lower().split()
    noise_free_words = [word for word in words if word not in stop] 
    noise_free_text = " ".join(noise_free_words) 
    return noise_free_text

df["clean_tweet_text"] = df["clean_tweet_text"].apply(lambda s: rem_en(s))
df[["tweet_text","clean_tweet_text"]]
#another defined term to clean the tweet_text column and show it next to the original


# In[12]:


from nltk.tokenize import RegexpTokenizer
tokeniser = RegexpTokenizer(r'\w+')
df["clean_tweet_text"] = df["clean_tweet_text"].apply(lambda x: tokeniser.tokenize(x))
df[["tweet_text","clean_tweet_text"]]


# In[ ]:





# In[13]:


texts= df["clean_tweet_text"]


texts
#list(myList[0])

words_list = [word for line in list(texts[0]) for word in line.split()]
words_list


# In[14]:


words_list[:5]


# In[15]:


from zipfile import ZipFile
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import _stop_words
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
import nltk
from nltk import word_tokenize
from nltk.stem.snowball import EnglishStemmer
import matplotlib.pylab as plt
from dmba import printTermDocumentMatrix, classificationSummary, liftChart

nltk.download('punkt')


# In[16]:


class LemmaTokenizer(object):
    def __init__(self):
        self.stemmer = EnglishStemmer()
        self.stopWords = set(ENGLISH_STOP_WORDS)
        
    def __call__(self, doc):
        return [self.stemmer.stem(t) for t in word_tokenize(doc) 
                if t.isalpha() and t not in self.stopWords]
#defines LemmaTokenizer, uses text analysis for this function, initializes english stemmer
#defines a call method and uses the nltk library
#stemming to each token applied


# In[17]:


count_vect = CountVectorizer(tokenizer=LemmaTokenizer())
counts = count_vect.fit_transform(list(texts[0]))

printTermDocumentMatrix(count_vect, counts)
#sets a countvecotrizes and through pre-processing the texts occurance numbers will be counted and displayed


# In[ ]:


#rest of the data did not work so I switched to another document!!!


# In[19]:


positive_words = ["amazing", "appreciate", "awesome", "beautiful", "best", "brilliant", "celebrate", "cheer", "cool",
                  "delicious", "eager", "enjoy", "fortunate", "fun", "glad", "good", "happy", "kind","love", "merry", "nice",
                  "pleasant", "polite", "praise", "relax", "Sweet", "top-notch", "win", "yay"]

negative_words = ["aggressive", "anger", "annoy", "bad", "bloody", "bored", "careless", "cocky", "death", "defy", "denial",
                  "detest", "dirty", "error", "fail", "guilt", "hate", "haunt", "idiot", "implode", "inhumane", "insult",
                  "irritate", "lousy", "mad", "outrage", "poor", "refute", "sad", "sick", "strict", "stuck", "unequal",
                  "waste", "wrong"]


# In[62]:


for tweet in df['tweet_text']:
    tokens = tweet.split()
    positive_words = []
    for token in tokens:
        if token.lower() in p_words:
            positive_words.append(token)
    print(positive_words)


# In[75]:


import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


# In[ ]:


sia = SentimentIntensityAnalyzer()


# In[70]:


positive_words = df['tweet_text'].apply(lambda x: [word for word in nltk.word_tokenize(x.lower()) if word.isalpha()])


# In[67]:


import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop = set(stopwords.words('english'))
print(stop)


# In[ ]:





# In[ ]:





# In[69]:





# In[20]:


text = df["tweet_text"]
tokeniser = RegexpTokenizer(r'\w+')


# In[63]:


#Remove stop words
stopWords = list(sorted(ENGLISH_STOP_WORDS))


# In[22]:


df.dropna(inplace=True)


# In[ ]:





# In[23]:


df['tweet_text'] = pd.to_numeric(df['tweet_text'], errors = 'coerce')


# In[ ]:





# In[ ]:





# In[ ]:





# In[24]:


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


# In[25]:


df = pd.read_csv('/Users/anthonyvincent/Downloads/iphone14-query-tweets.csv.zip')
df.head()


# In[26]:


df.info()


# In[27]:


df.dropna(inplace=True)


# In[28]:


#1,2,3->negative(i.e 0)
df.loc[df['tweet_like_count'] <= 3,'tweet_like_count'] = 0
 
#4,5->positive(i.e 1)
df.loc[df['tweet_like_count'] > 3,'tweet_like_count'] = 1


# In[29]:


stp_words=stopwords.words('english')

def clean_review(review):
    if isinstance(review, str):
        cleanreview=" ".join(word for word in review.split() if word not in stp_words)
        return cleanreview
    else:
        return review 
    
df['tweet_text']=df['tweet_text'].apply(clean_review)


# In[30]:


df.head()


# In[31]:


#HELPFUL INFORMATION ON THE NUMBER OF LIKES A TWEET GETS


# In[32]:


df['tweet_like_count'].value_counts()


# In[33]:


df['tweet_retweet_count'].value_counts()


# In[34]:


df['tweet_text'].value_counts()


# tweets that are liked by customers show a positve correlation or meaningful message?

# In[37]:


consolidated=' '.join(word for word in df['tweet_text'][df['tweet_like_count']==0].astype(str))
wordCloud=WordCloud(width=1600,height=800,random_state=21,max_font_size=110)
plt.figure(figsize=(15,10))
plt.imshow(wordCloud.generate(consolidated),interpolation='bilinear')
plt.axis('off')
plt.show()


# In[38]:


consolidated=' '.join(word for word in df['tweet_text'][df['tweet_like_count']==1].astype(str))
wordCloud=WordCloud(width=1600,height=800,random_state=21,max_font_size=110)
plt.figure(figsize=(15,10))
plt.imshow(wordCloud.generate(consolidated),interpolation='bilinear')
plt.axis('off')
plt.show()


# In[ ]:


why no image???


# TF-IDF calculates that how relevant a word in a series or corpus is to a text. The meaning increases proportionally to the number of times in the text a word appears but is compensated by the word frequency in the corpus (data-set). We will be implementing this with the code below.

# In[ ]:


df['tweet_text'] = df['tweet_text'].astype(str)


# In[ ]:


cv = TfidfVectorizer(max_features=2500)
X = cv.fit_transform(df['tweet_text'] ).toarray()


# In[ ]:


from sklearn.model_selection import train_test_split
x_train ,x_test,y_train,y_test=train_test_split(X,df['tweet_like_count'],
                                                test_size=0.25 ,
                                                random_state=42)


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
model=LogisticRegression()
 
#Model fitting
model.fit(x_train,y_train)
 
#testing the model
pred=model.predict(x_test)
 
#model accuracy
print(accuracy_score(y_test,pred))


# In[ ]:


from sklearn import metrics
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,pred)
 
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm,
                                            display_labels = [False, True])
 
cm_display.plot()
plt.show()


# In[ ]:


why no image???


# In[ ]:





# POSITIVE AND NEGATIVE WORDS

# In[64]:


positive_words = []
with open('positive_words.txt', 'r') as f:
    for line in f 
    positive_words.append(line.strip)
    
def extract_positve_words(tweet_text)


# In[58]:


df["tweet_text"].value_counts()


# In[ ]:




