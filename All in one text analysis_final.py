# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 18:02:40 2016

@author: Sayantan Das (stat.sayantan@gmail.com)
This python program creates a word cloud of any shape and of any font of any color and pattern (i.e. percent of words displayed horizontally) from excel data file. 
It also creates desired number of topics with desired number of words for each topic.
It also outputs the sentiment and subjectivity scores and classifies each sentence into one of the topics.
Additionally, it creates a summary report in the same excel and displays frequencies of words in decreasing order under each topic.
"""

import pandas as pd
from wordcloud import WordCloud
from nltk.corpus import stopwords
import re
import string
import os
from collections import Counter
import random
from PIL import Image
import numpy as np
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from textblob import TextBlob
from sklearn.cluster import KMeans


#Ensures that the named directory exists
def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

#These are all color functions used to generate the wordcloud in different shades
def vivid_color_func(word, font_size, position, orientation, random_state=None,**kwargs):
    return "hsl(%d, %d%%, %d%%)" % (random.sample((random.randint(1,7),random.randint(18,23),random.randint(49,54),random.randint(63,68),random.randint(98,103),random.randint(165,170),random.randint(233,238),random.randint(312,318),random.randint(351,357)),1)[0],random.randint(95,100),random.randint(48,52))

def grey_color_func(word, font_size, position, orientation, random_state=None,**kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(30, 90)
    
def red_color_func(word, font_size, position, orientation, random_state=None,**kwargs):
    return "hsl(0, 99%%, %d%%)" % random.randint(30, 90)

def blue_color_func(word, font_size, position, orientation, random_state=None,**kwargs):
    return "hsl(247, 99%%, %d%%)" % random.randint(30, 90)
    
def yellow_color_func(word, font_size, position, orientation, random_state=None,**kwargs):
    return "hsl(55, 99%%, %d%%)" % random.randint(30, 90)

def green_color_func(word, font_size, position, orientation, random_state=None,**kwargs):
    return "hsl(112, 99%%, %d%%)" % random.randint(30, 90)

def cyan_color_func(word, font_size, position, orientation, random_state=None,**kwargs):
    return "hsl(168, 99%%, %d%%)" % random.randint(30, 90)
    
#This function generates the wordcloud as specified by the user and saves it in the path same as instruction file
def savecloud(location,cloudtype,mask,run=0,width=800,height=600,margin=0,prefer_horizontal=0.9,font=None):
    if run==0:
        if int(cloudtype)==1:
            wordcloud = WordCloud(color_func=vivid_color_func,background_color="white",width=width,height=height,margin=margin,prefer_horizontal=prefer_horizontal,font_path=font).generate_from_frequencies([tuple(x) for x in word_freq.values])
        elif int(cloudtype)==2:
            wordcloud = WordCloud(color_func=vivid_color_func,background_color="black",width=width,height=height,margin=margin,prefer_horizontal=prefer_horizontal,font_path=font).generate_from_frequencies([tuple(x) for x in word_freq.values])
        elif int(cloudtype)==3:
            wordcloud = WordCloud(color_func=grey_color_func,width=width,height=height,margin=margin,prefer_horizontal=prefer_horizontal,font_path=font).generate_from_frequencies([tuple(x) for x in word_freq.values])
        elif int(cloudtype)==4:
            wordcloud = WordCloud(color_func=red_color_func,width=width,height=height,margin=margin,prefer_horizontal=prefer_horizontal,font_path=font).generate_from_frequencies([tuple(x) for x in word_freq.values])
        elif int(cloudtype)==5:
            wordcloud = WordCloud(color_func=blue_color_func,width=width,height=height,margin=margin,prefer_horizontal=prefer_horizontal,font_path=font).generate_from_frequencies([tuple(x) for x in word_freq.values])
        elif int(cloudtype)==6:
            wordcloud = WordCloud(color_func=yellow_color_func,width=width,height=height,margin=margin,prefer_horizontal=prefer_horizontal,font_path=font).generate_from_frequencies([tuple(x) for x in word_freq.values])
        elif int(cloudtype)==7:
            wordcloud = WordCloud(color_func=green_color_func,width=width,height=height,margin=margin,prefer_horizontal=prefer_horizontal,font_path=font).generate_from_frequencies([tuple(x) for x in word_freq.values])
        elif int(cloudtype)==8:
            wordcloud = WordCloud(color_func=cyan_color_func,width=width,height=height,margin=margin,prefer_horizontal=prefer_horizontal,font_path=font).generate_from_frequencies([tuple(x) for x in word_freq.values])
        elif int(cloudtype)==9:
            wordcloud = WordCloud(color_func=grey_color_func,background_color="white",width=width,height=height,margin=margin,prefer_horizontal=prefer_horizontal,font_path=font).generate_from_frequencies([tuple(x) for x in word_freq.values])
        elif int(cloudtype)==10:
            wordcloud = WordCloud(color_func=red_color_func,background_color="white",width=width,height=height,margin=margin,prefer_horizontal=prefer_horizontal,font_path=font).generate_from_frequencies([tuple(x) for x in word_freq.values])
        elif int(cloudtype)==11:
            wordcloud = WordCloud(color_func=blue_color_func,background_color="white",width=width,height=height,margin=margin,prefer_horizontal=prefer_horizontal,font_path=font).generate_from_frequencies([tuple(x) for x in word_freq.values])
        elif int(cloudtype)==12:
            wordcloud = WordCloud(color_func=yellow_color_func,background_color="white",width=width,height=height,margin=margin,prefer_horizontal=prefer_horizontal,font_path=font).generate_from_frequencies([tuple(x) for x in word_freq.values])
        elif int(cloudtype)==13:
            wordcloud = WordCloud(color_func=green_color_func,background_color="white",width=width,height=height,margin=margin,prefer_horizontal=prefer_horizontal,font_path=font).generate_from_frequencies([tuple(x) for x in word_freq.values])
        elif int(cloudtype)==14:
            wordcloud = WordCloud(color_func=cyan_color_func,background_color="white",width=width,height=height,margin=margin,prefer_horizontal=prefer_horizontal,font_path=font).generate_from_frequencies([tuple(x) for x in word_freq.values])
    else:
        if int(cloudtype)==1:
            wordcloud = WordCloud(background_color="white",width=width,height=height,margin=margin,mask=demomask,prefer_horizontal=prefer_horizontal,font_path=font).generate_from_frequencies([tuple(x) for x in word_freq.values])
        elif int(cloudtype)==2:
            wordcloud = WordCloud(background_color="black",width=width,height=height,margin=margin,mask=demomask,prefer_horizontal=prefer_horizontal,font_path=font).generate_from_frequencies([tuple(x) for x in word_freq.values])
        elif int(cloudtype)==3:
            wordcloud = WordCloud(color_func=grey_color_func,width=width,height=height,margin=margin,mask=demomask,prefer_horizontal=prefer_horizontal,font_path=font).generate_from_frequencies([tuple(x) for x in word_freq.values])
        elif int(cloudtype)==4:
            wordcloud = WordCloud(color_func=red_color_func,width=width,height=height,margin=margin,mask=demomask,prefer_horizontal=prefer_horizontal,font_path=font).generate_from_frequencies([tuple(x) for x in word_freq.values])
        elif int(cloudtype)==5:
            wordcloud = WordCloud(color_func=blue_color_func,width=width,height=height,margin=margin,mask=demomask,prefer_horizontal=prefer_horizontal,font_path=font).generate_from_frequencies([tuple(x) for x in word_freq.values])
        elif int(cloudtype)==6:
            wordcloud = WordCloud(color_func=yellow_color_func,width=width,height=height,margin=margin,mask=demomask,prefer_horizontal=prefer_horizontal,font_path=font).generate_from_frequencies([tuple(x) for x in word_freq.values])
        elif int(cloudtype)==7:
            wordcloud = WordCloud(color_func=green_color_func,width=width,height=height,margin=margin,mask=demomask,prefer_horizontal=prefer_horizontal,font_path=font).generate_from_frequencies([tuple(x) for x in word_freq.values])
        elif int(cloudtype)==8:
            wordcloud = WordCloud(color_func=cyan_color_func,width=width,height=height,margin=margin,mask=demomask,prefer_horizontal=prefer_horizontal,font_path=font).generate_from_frequencies([tuple(x) for x in word_freq.values])
        elif int(cloudtype)==9:
            wordcloud = WordCloud(color_func=grey_color_func,background_color="white",width=width,height=height,margin=margin,mask=demomask,prefer_horizontal=prefer_horizontal,font_path=font).generate_from_frequencies([tuple(x) for x in word_freq.values])
        elif int(cloudtype)==10:
            wordcloud = WordCloud(color_func=red_color_func,background_color="white",width=width,height=height,margin=margin,mask=demomask,prefer_horizontal=prefer_horizontal,font_path=font).generate_from_frequencies([tuple(x) for x in word_freq.values])
        elif int(cloudtype)==11:
            wordcloud = WordCloud(color_func=blue_color_func,background_color="white",width=width,height=height,margin=margin,mask=demomask,prefer_horizontal=prefer_horizontal,font_path=font).generate_from_frequencies([tuple(x) for x in word_freq.values])
        elif int(cloudtype)==12:
            wordcloud = WordCloud(color_func=yellow_color_func,background_color="white",width=width,height=height,margin=margin,mask=demomask,prefer_horizontal=prefer_horizontal,font_path=font).generate_from_frequencies([tuple(x) for x in word_freq.values])
        elif int(cloudtype)==13:
            wordcloud = WordCloud(color_func=green_color_func,background_color="white",width=width,height=height,margin=margin,mask=demomask,prefer_horizontal=prefer_horizontal,font_path=font).generate_from_frequencies([tuple(x) for x in word_freq.values])
        elif int(cloudtype)==14:
            wordcloud = WordCloud(color_func=cyan_color_func,background_color="white",width=width,height=height,margin=margin,mask=demomask,prefer_horizontal=prefer_horizontal,font_path=font).generate_from_frequencies([tuple(x) for x in word_freq.values])
    wordcloud.to_file(location)
        
        
#Running automated sentiment analysis
def run_sentiment_automated(data):
    data.reset_index(drop=True,inplace=True)
    temp = data.apply(lambda x: TextBlob(x).sentiment).reset_index(drop=True)
    data = pd.concat([data,temp.apply(lambda x: x.polarity)], axis=1).reset_index(drop=True)
    data.columns = [data.columns[0], 'Sentiment score']
    conditions = [
    (data['Sentiment score'] > 0.5),
    (data['Sentiment score'] < -0.5)]
    choices = ['Positive', 'Negative']
    data['Sentiment'] = np.select(conditions, choices, default='Neutral Sentiment')
    data['Subjectivity score'] = temp.apply(lambda x: x.subjectivity)
    conditions = [
    (data['Subjectivity score'] < 0.25),
    (data['Subjectivity score'] > 0.75)]
    choices = ['Objective', 'Subjective']
    data['Subjectivity'] = np.select(conditions, choices, default='Neutral Subjectivity')
    data.reset_index(drop=True, inplace=True)
    return(data[['Sentiment','Subjectivity','Sentiment score','Subjectivity score']])        
		
def print_top_words(model, feature_names, n_top_words):
    combined_messages = pd.DataFrame(columns = ("Topic","Words"))
    n=0
    for topic_idx, topic in enumerate(model.components_):
        combined_messages.loc[n,"Topic"] = "Topic %d" % (topic_idx)
        combined_messages.loc[n,"Words"] = " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        n+= 1
    return(combined_messages)

def get_percent_topic(model,tfidf,typemodel = 'lda'):
    #Ignoring error due to divide by zero and divide by 'nan'
    np.seterr(divide='ignore', invalid='ignore')
    data_score = pd.DataFrame(model.transform(tfidf))
    data_score.columns = ['Topic ' + str(col) for col in data_score.columns]
    data_score['Belongs to'] = data_score.idxmax(axis=1)
    score_var = [col for col in data_score.columns if ('Topic' in col)]
    data_score['First'] = data_score[score_var].apply(lambda row: row.nlargest(1).values[-0],axis=1)
    data_score['Second'] = data_score[score_var].apply(lambda row: row.nlargest(2).values[-1],axis=1)
    data_score['Denominator'] = np.sqrt((((data_score['First'] + data_score['Second'])/2)*(1-((data_score['First'] + data_score['Second'])/2)))/50)
    data_score['P observed'] = ((data_score['First'] - data_score['Second'])/data_score['Denominator'])
    if typemodel == 'lda':
        data_score['Condition'] = data_score['P observed']  > 2.58
    #else:
    #    data_score['Condition'] = data_score['P observed']  > 1.645
        data_score.loc[data_score['Condition'] == False,'Belongs to'] = str('Not significant')
    return(data_score['Belongs to'])
    

def get_topic_diagram(data1,data2):
    finaldata = pd.DataFrame()
    for i in data1['Topic']:
        text1 = data2.loc[data2['Belongs to']==i,data2.columns[0]].str.split(expand=True).stack().value_counts()
        text1 = text1.reset_index(name='Freq')
        text1 = text1.rename(columns={'index': 'Words'})
        addsuffix = '_' + i
        text1.columns = [str(col) + addsuffix for col in text1.columns]
        text1.reset_index(inplace=True,drop=True)
        if finaldata.shape[0] == 0:
            finaldata = text1.copy()
        else:
            finaldata = pd.concat([finaldata.reset_index(drop=True), text1.reset_index(drop=True)], axis=1)
    return(finaldata)

cList = {
  "ain't": "am not",
  "aren't": "are not",
  "can't": "cannot",
  "can't've": "cannot have",
  "'cause": "because",
  "could've": "could have",
  "couldn't": "could not",
  "couldn't've": "could not have",
  "didn't": "did not",
  "doesn't": "does not",
  "don't": "do not",
  "hadn't": "had not",
  "hadn't've": "had not have",
  "hasn't": "has not",
  "haven't": "have not",
  "he'd": "he would",
  "he'd've": "he would have",
  "he'll": "he will",
  "he'll've": "he will have",
  "he's": "he is",
  "how'd": "how did",
  "how'd'y": "how do you",
  "how'll": "how will",
  "how's": "how is",
  "i'd": "i would",
  "i'd've": "i would have",
  "i'll": "i will",
  "i'll've": "i will have",
  "i'm": "i am",
  "i've": "i have",
  "isn't": "is not",
  "it'd": "it had",
  "it'd've": "it would have",
  "it'll": "it will",
  "it'll've": "it will have",
  "it's": "it is",
  "let's": "let us",
  "ma'am": "madam",
  "mayn't": "may not",
  "might've": "might have",
  "mightn't": "might not",
  "mightn't've": "might not have",
  "must've": "must have",
  "mustn't": "must not",
  "mustn't've": "must not have",
  "needn't": "need not",
  "needn't've": "need not have",
  "o'clock": "of the clock",
  "oughtn't": "ought not",
  "oughtn't've": "ought not have",
  "shan't": "shall not",
  "sha'n't": "shall not",
  "shan't've": "shall not have",
  "she'd": "she would",
  "she'd've": "she would have",
  "she'll": "she will",
  "she'll've": "she will have",
  "she's": "she is",
  "should've": "should have",
  "shouldn't": "should not",
  "shouldn't've": "should not have",
  "so've": "so have",
  "so's": "so is",
  "that'd": "that would",
  "that'd've": "that would have",
  "that's": "that is",
  "there'd": "there had",
  "there'd've": "there would have",
  "there's": "there is",
  "they'd": "they would",
  "they'd've": "they would have",
  "they'll": "they will",
  "they'll've": "they will have",
  "they're": "they are",
  "they've": "they have",
  "to've": "to have",
  "wasn't": "was not",
  "we'd": "we had",
  "we'd've": "we would have",
  "we'll": "we will",
  "we'll've": "we will have",
  "we're": "we are",
  "we've": "we have",
  "weren't": "were not",
  "what'll": "what will",
  "what'll've": "what will have",
  "what're": "what are",
  "what's": "what is",
  "what've": "what have",
  "when's": "when is",
  "when've": "when have",
  "where'd": "where did",
  "where's": "where is",
  "where've": "where have",
  "who'll": "who will",
  "who'll've": "who will have",
  "who's": "who is",
  "who've": "who have",
  "why's": "why is",
  "why've": "why have",
  "will've": "will have",
  "won't": "will not",
  "won't've": "will not have",
  "would've": "would have",
  "wouldn't": "would not",
  "wouldn't've": "would not have",
  "y'all": "you all",
  "y'alls": "you alls",
  "y'all'd": "you all would",
  "y'all'd've": "you all would have",
  "y'all're": "you all are",
  "y'all've": "you all have",
  "you'd": "you had",
  "you'd've": "you would have",
  "you'll": "you you will",
  "you'll've": "you you will have",
  "you're": "you are",
  "you've": "you have",
  "was": "is",
  "has": "have"
}

c_re = re.compile('(%s)' % '|'.join(cList.keys()))

def expandContractions(text, c_re=c_re):
    def replace(match):
        return cList[match.group(0)]
    return(c_re.sub(replace, text))
    
#Function to clean data    
def clean_data(wordle_data,output_form=1):

    wordle_data = wordle_data.apply(lambda x: x.replace("nan",""))
	#Make the words lowercase
	wordle_data = wordle_data.apply(lambda x: x.lower())

	#Replacing custom string        
	if len(rep)>0:
		pattern = re.compile("|".join(rep.keys()))
		wordle_data = wordle_data.apply(lambda x: str(pattern.sub(lambda m: rep[re.escape(m.group(0))], x)))
	
	#Replacing pre-specified patterns
	wordle_data = wordle_data.apply(lambda x: expandContractions(x))     
	
	#Removing apostrophe s
	wordle_data = wordle_data.apply(lambda x: re.sub("'s? ", " ", x))
	
    num_re = re.compile('(\\d+)')
    # remove numeric 'words'
    wordle_data = wordle_data.apply(lambda x: num_re.sub(' ', x))
    mention_re = re.compile('@(\w+)')
    # remove @mentions
    wordle_data = wordle_data.apply(lambda x: mention_re.sub(' ', x))
    http_re = re.compile(r"(?:\@|https?\://)\S+")
    www_re = re.compile(r"(?:\@|www?\://)\S+")
    # remove url's
    wordle_data = wordle_data.apply(lambda x: http_re.sub(' ', x))
    wordle_data = wordle_data.apply(lambda x: www_re.sub(' ', x))
    punc_re = re.compile('[%s]' % re.escape(string.punctuation))

    # remove puncuation
    wordle_data = wordle_data.apply(lambda x: punc_re.sub(' ', x))

    # Lemmatization
    wordle_data = wordle_data.apply(lambda x: ' '.join([wordnet_lemmatizer.lemmatize(word,'v') for word in str(x).split()]))
    wordle_data = wordle_data.apply(lambda x: ' '.join([wordnet_lemmatizer.lemmatize(word) for word in str(x).split()]))
    
    #Remove all stop-words like I,me,my,we,our etc.
    cachedStopWords = stopwords.words("english")
    wordle_data = wordle_data.apply(lambda x: ' '.join([word for word in x.split() if word not in cachedStopWords]))
    if output_form == 2:
        wordle_data = ' '.join(str(e) for e in wordle_data)
    return(wordle_data)
        
#Runs topic modeling on the data
def run_topic(data,n_components,n_top_words,file_path):
    
    cleandata = clean_data(data,1)
    # Use tf-idf features for NMF.
    #print("Extracting tf-idf features for NMF...")
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,stop_words='english')
    #t0 = time()
    tfidf = tfidf_vectorizer.fit_transform(cleandata)
    #print("done in %0.3fs." % (time() - t0))
    
    # Use tf (raw term count) features for LDA.
    #print("Extracting tf features for LDA...")
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,stop_words='english')
    #t0 = time()
    tf = tf_vectorizer.fit_transform(cleandata)
    #print("done in %0.3fs." % (time() - t0))
    #print()
    
    # Fit the NMF model
    #print("Fitting the NMF model (Frobenius norm) with tf-idf features, ""n_samples=%d and n_features=%d..." % (n_samples, n_features))
    #t0 = time()
    sentimentdata = run_sentiment_automated(cleandata)
    #Finding out the details for each review
    nmf = NMF(n_components=n_components, random_state=1,alpha=.1, l1_ratio=.5).fit(tfidf)
    topic_absolute = pd.concat([cleandata,get_percent_topic(nmf,tfidf,'nmf')],axis=1)
    #topic_absolute.reset_index(drop=True, inplace=True)
    nmf1_topic_absolute = pd.concat([topic_absolute,sentimentdata],axis=1).reset_index(drop=True)
    #nmf1_topic_absolute = nmf1_topic_absolute.sort_values(by='Belongs to',ascending=True)
    topic_sentiment = pd.DataFrame(pd.pivot_table(nmf1_topic_absolute[['Belongs to','Sentiment']],index=['Belongs to'],columns=['Sentiment'],aggfunc=len).fillna(0).reset_index())
    topic_subjectivity = pd.DataFrame(pd.pivot_table(nmf1_topic_absolute[['Belongs to','Subjectivity']],index=['Belongs to'],columns=['Subjectivity'],aggfunc=len).fillna(0).reset_index())
    topic_count = pd.DataFrame(nmf1_topic_absolute['Belongs to'].value_counts())
    topic_count.reset_index(inplace=True)
    topic_count = topic_count.rename(columns={'index': 'Belongs to', 'Belongs to': 'Count'})
    topic_gist = pd.merge(left=topic_sentiment,right=topic_subjectivity, left_on='Belongs to', right_on='Belongs to')
    nmf1_topic_gist = pd.merge(left=topic_gist,right=topic_count, left_on='Belongs to', right_on='Belongs to')
    #print("done in %0.3fs." % (time() - t0))
    
    #print("\nTopics in NMF model (Frobenius norm):")
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    top_words_nmf_Frobenius = print_top_words(nmf, tfidf_feature_names, n_top_words)
    mainwords_nmf1 = get_topic_diagram(top_words_nmf_Frobenius,nmf1_topic_absolute)
    nmf1_topic_absolute[nmf1_topic_absolute.columns[0]] = data
    
    # Fit the NMF model
    #print("Fitting the NMF model (generalized Kullback-Leibler divergence) with ""tf-idf features, n_samples=%d and n_features=%d..."% (n_samples, n_features))
    #t0 = time()
    nmf = NMF(n_components=n_components, random_state=1,beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1,l1_ratio=.5).fit(tfidf)
    topic_absolute = pd.concat([cleandata,get_percent_topic(nmf,tfidf,'nmf')],axis=1)
    #topic_absolute.reset_index(drop=True, inplace=True)
    nmf2_topic_absolute = pd.concat([topic_absolute,sentimentdata],axis=1)
    #nmf2_topic_absolute = nmf2_topic_absolute.sort_values(by='Belongs to',ascending=True)
    topic_sentiment = pd.DataFrame(pd.pivot_table(nmf2_topic_absolute[['Belongs to','Sentiment']],index=['Belongs to'],columns=['Sentiment'],aggfunc=len).fillna(0).reset_index())
    topic_subjectivity = pd.DataFrame(pd.pivot_table(nmf2_topic_absolute[['Belongs to','Subjectivity']],index=['Belongs to'],columns=['Subjectivity'],aggfunc=len).fillna(0).reset_index())
    topic_count = pd.DataFrame(nmf2_topic_absolute['Belongs to'].value_counts())
    topic_count.reset_index(inplace=True)
    topic_count = topic_count.rename(columns={'index': 'Belongs to', 'Belongs to': 'Count'})
    topic_gist = pd.merge(left=topic_sentiment,right=topic_subjectivity, left_on='Belongs to', right_on='Belongs to')
    nmf2_topic_gist = pd.merge(left=topic_gist,right=topic_count, left_on='Belongs to', right_on='Belongs to')

    #print("done in %0.3fs." % (time() - t0))
    
    #print("\nTopics in NMF model (generalized Kullback-Leibler divergence):")
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    top_words_nmf_KLdivergence = print_top_words(nmf, tfidf_feature_names, n_top_words)
    mainwords_nmf2 = get_topic_diagram(top_words_nmf_KLdivergence,nmf2_topic_absolute)
    nmf2_topic_absolute[nmf2_topic_absolute.columns[0]] = data
    
    #print("Fitting LDA models with tf features, ""n_samples=%d and n_features=%d..."% (n_samples, n_features))
    lda = LatentDirichletAllocation(n_components=n_components, max_iter=5,learning_method='online',learning_offset=50.,random_state=0)
    #t0 = time()
    lda.fit(tf)
    #print("done in %0.3fs." % (time() - t0))
    topic_absolute = pd.concat([cleandata,get_percent_topic(lda,tfidf)],axis=1)
    #topic_absolute.reset_index(drop=True, inplace=True)
    lda_topic_absolute = pd.concat([topic_absolute,sentimentdata],axis=1)
    #lda_topic_absolute = lda_topic_absolute.sort_values(by='Belongs to',ascending=True)
    topic_sentiment = pd.DataFrame(pd.pivot_table(lda_topic_absolute[['Belongs to','Sentiment']],index=['Belongs to'],columns=['Sentiment'],aggfunc=len).fillna(0).reset_index())
    topic_subjectivity = pd.DataFrame(pd.pivot_table(lda_topic_absolute[['Belongs to','Subjectivity']],index=['Belongs to'],columns=['Subjectivity'],aggfunc=len).fillna(0).reset_index())
    topic_count = pd.DataFrame(lda_topic_absolute['Belongs to'].value_counts())
    topic_count.reset_index(inplace=True)
    topic_count = topic_count.rename(columns={'index': 'Belongs to', 'Belongs to': 'Count'})
    topic_gist = pd.merge(left=topic_sentiment,right=topic_subjectivity, left_on='Belongs to', right_on='Belongs to')
    lda_topic_gist = pd.merge(left=topic_gist,right=topic_count, left_on='Belongs to', right_on='Belongs to')

    #print("\nTopics in LDA model:")
    tf_feature_names = tf_vectorizer.get_feature_names()
    top_words_lda = print_top_words(lda, tf_feature_names, n_top_words)
    mainwords_lda = get_topic_diagram(top_words_lda,lda_topic_absolute)
    lda_topic_absolute[lda_topic_absolute.columns[0]] = data
    
    #K-means clustering
    model = KMeans(n_clusters=n_components, init='k-means++', max_iter=100, n_init=1)
    model.fit(tfidf)

    #print("Top terms per cluster:")
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = tfidf_vectorizer.get_feature_names()
    cluster_topic = pd.DataFrame(columns=['Topic','Words'])
    n=0
    for i in range(n_components):
        cluster_topic.loc[n,'Topic'] = "Topic %d" %i
        temp = list()
        for ind in order_centroids[i, :n_top_words]:
            temp.append(terms[ind])
        temp = " ".join(map(str, temp))
        cluster_topic.loc[n,'Words'] = temp
        n+=1
    cluster_topic_absolute = pd.Series(model.labels_.tolist())
    cluster_topic_absolute = cluster_topic_absolute.to_frame('Belongs to')
    cluster_topic_absolute = pd.concat([cleandata,cluster_topic_absolute],axis=1)
    cluster_topic_absolute['Belongs to'] = 'Topic ' + cluster_topic_absolute['Belongs to'].astype(str)
    cluster_topic_absolute = pd.concat([cluster_topic_absolute,sentimentdata],axis=1).reset_index(drop=True)
    topic_sentiment = pd.DataFrame(pd.pivot_table(cluster_topic_absolute[['Belongs to','Sentiment']],index=['Belongs to'],columns=['Sentiment'],aggfunc=len).fillna(0).reset_index())
    topic_subjectivity = pd.DataFrame(pd.pivot_table(cluster_topic_absolute[['Belongs to','Subjectivity']],index=['Belongs to'],columns=['Subjectivity'],aggfunc=len).fillna(0).reset_index())
    topic_count = pd.DataFrame(cluster_topic_absolute['Belongs to'].value_counts())
    topic_count.reset_index(inplace=True)
    topic_count = topic_count.rename(columns={'index': 'Belongs to', 'Belongs to': 'Count'})
    topic_gist = pd.merge(left=topic_sentiment,right=topic_subjectivity, left_on='Belongs to', right_on='Belongs to')
    kmeans_topic_gist = pd.merge(left=topic_gist,right=topic_count, left_on='Belongs to', right_on='Belongs to') 
    mainwords_kmeans = get_topic_diagram(cluster_topic,cluster_topic_absolute)
    cluster_topic_absolute[cluster_topic_absolute.columns[0]] = data

    # Writing out the topics    
    writer = pd.ExcelWriter(file_path)
    top_words_lda.to_excel(writer,'lda',index=False)
    lda_topic_absolute.to_excel(writer,'lda_details',index=False)
    lda_topic_gist.to_excel(writer,'lda_gist',index=False)
    mainwords_lda.to_excel(writer,'lda_keywords',index=False)
    cluster_topic.to_excel(writer,'Kmeans',index=False)
    cluster_topic_absolute.to_excel(writer,'Kmeans_details',index=False)
    kmeans_topic_gist.to_excel(writer,'Kmeans_gist',index=False)
    mainwords_kmeans.to_excel(writer,'Kmeans_keywords',index=False)
    top_words_nmf_Frobenius.to_excel(writer,'nmf_Frobenius',index=False)
    nmf1_topic_absolute.to_excel(writer,'nmf_Frobenius_details',index=False)
    nmf1_topic_gist.to_excel(writer,'nmf_Frobenius_gist',index=False)
    mainwords_nmf1.to_excel(writer,'nmf_Frob_keywords',index=False)
    top_words_nmf_KLdivergence.to_excel(writer,'nmf_KLdivergence',index=False)
    nmf2_topic_absolute.to_excel(writer,'nmf_KLdiv_details',index=False)
    nmf2_topic_gist.to_excel(writer,'nmf_KLdiv_gist',index=False)
    mainwords_nmf2.to_excel(writer,'nmf_KLdiv_keywords',index=False)
    writer.save()

  
#Reads the information/instruction excel file 
path=str()
while path.endswith(".xlsx")==False:
    path=input("Instruction file location (viz. c:/Eg/Instructions.xlsx): ")
path=path.replace("\\","/")
folder='/'.join(path.split('/')[0:-1])
info=pd.ExcelFile(path)
info=info.parse()
wordnet_lemmatizer = WordNetLemmatizer()
    
# Take information about topic modeling
n_components=str()
while (not isinstance(n_components, int)):
    n_components=input("Please enter number of topics you want to extract: ")
    try:
        n_components = int(n_components)
    except:
        pass

n_top_words=str()
while (not isinstance(n_top_words, int)):
    n_top_words=input("Please enter number of words for each topic you want to see: ")
    try:
        n_top_words = int(n_top_words)
    except:
        pass	
print("\n***This program has an inbuild sentiment analysis with topic modelling that uses the library 'textblob' to calculate sentiment and subjectivity. This comes with a model trained on movie reviews training data, hence use with caution\n")

#if sentiment_required == 'y':
#    auto_sentiment=str()
#    while auto_sentiment=="":
#        auto_sentiment=input("Do you have training data for sentiment analysis [y/n]: ")
#        if auto_sentiment == 'n':
#            print("\nThe program will try to run an automated sentiment analysis\n")
#        elif auto_sentiment == 'y':
#            print("\nThe training file for sentiment anaysis needs to have only two columns, text and sentiment with column header\n")
#            training_data = input("Sentiment training file location (viz. c:/Eg/Instructions.xlsx): ")
#
    
# Taking input from user if they want to include two or three words in the wordle
print()
print("If there are several multi-words like 'ice cream' appearing in the text, the script will automatically try to include them as multi words in the wordle.")
print()
want_multi_word = input("Do you wish to include multi-words in wordle? [y/n]: ")
if want_multi_word == 'y':
    cutoff = str()
    while (not isinstance(cutoff, int)):
        cutoff=input("Please enter a cut-off frequency to consider for such multi words: ")
        try:
            cutoff = int(cutoff)
        except:
            pass


for index, row in info.iterrows():
    #This part reads and processes each row of the instruction file and creates the corresponding wordcloud
    #There are several if-else conditions here to tackle blank information cells
    data = row[0]
    data = data.replace("\\","/")
    if str(row[4])=='nan':
        selection = 500
    else:
        selection = row[4]
    if str(row[5])!='nan':
        replace_this = row[5].split(",")
    if str(row[6])!='nan':
        replace_with = row[6].split(",")
    if str(row[7])!='nan':
        cloudtype = str(row[7]).split(",")
    else:
        print("Please specify the required Chart_Type")
        #If there is no data for chart type, it will prompt the user and end the program
        break
    if str(row[8])!='nan':
        orientation = str(row[8]).split(",")
    else:
        print("Please specify Horizontal column")
        #If there is no data for percent of horizontal words, it will prompt the user and end the program
        break
    if str(row[9])!='nan':
        masktype = row[9].split(",")
    if str(row[10])!='nan':
        masklocation = row[10]
        masklocation = masklocation.replace("\\","/")
        demomask = np.array(Image.open(str(masklocation)))
    else:
        demomask = str('None')
    if str(row[11])!='nan':
        fontpath = row[11]
        fontpath = fontpath.replace("\\","/")
    else:
        fontpath = None
    if str(row[6])!='nan':
        rep = dict(zip(list(map((lambda x: x.lower()),replace_this)),list(map((lambda x: x.lower()),replace_with))))
    else:
        rep = dict()
    data = pd.ExcelFile(data)
    data = data.parse()
    n=-1
    if row[1].lower() != "total":
        for items in row[1].split(","):
            temp = data[data[data.dtypes.index[int(row[2])-1]]==items]
            n+=1
            if n==0:
                final=temp
            else:
                final=final.append(temp)
        data=final
    if str(row[3])!='nan':
        wordle_column = str(row[3]).split(",")
    #If for a row, the length of chart type,orientation percentage and masktype does not match the number of wordles to be created, the program will terminate with a user prompt
    if len(wordle_column) != len(cloudtype):
        print("Please rectify the input for Chart_Type; program will terminate")
        break
    elif len(wordle_column) != len(orientation):
        print("Please rectify the input for Horizontal; program will terminate")
        break
    elif len(wordle_column) != len(masktype):
        print("Please rectify the input for Mask; program will terminate")  
        break
    n=0
    for item in wordle_column:
        wordle_data = data.ix[:,int(item)-1]
        wordle_data = wordle_data.dropna()        
        
        # Creating a dataset for topic modelling
        data_for_topic = wordle_data.copy().dropna()
        
        # Getting clean data
        temp = clean_data(wordle_data,1)
        wordle_data = clean_data(wordle_data,2)
        
        #Creating dataframe of single words and their frequencies
        word_freq = pd.DataFrame(Counter(wordle_data.split()).most_common(selection))
        word_freq.columns = ("Word","Frequency")
                
        # Finding common occuring two and three words
        if want_multi_word == 'y':
            #words = re.findall(r'\w+', wordle_data)
            words = temp.apply(lambda x: re.findall(r'\w+', x))
            two_word_freq = words.apply(lambda x: [' '.join(ws) for ws in zip(x, x[1:])])
            two_word_freq = ''.join(str(e) for e in two_word_freq)
            two_word_freq = two_word_freq.replace("] [",", ").replace("[","").replace("]","").replace("'","").replace(", ",",").split(",")
            #two_word_freq = [' '.join(ws) for ws in zip(words, words[1:])]
            wordscount_2 = {w:f for w, f in Counter(two_word_freq).most_common() if f > (cutoff)}
            if len(wordscount_2) > 0:
                wordscount_2 = pd.DataFrame({"Word":list(wordscount_2.keys()),"Frequency":list(wordscount_2.values())})
                # This part reduces the frequency of single words by the frequency of its double word version
                for index1, row1 in wordscount_2.iterrows():
                    term = wordscount_2.loc[index1,'Word']
                    freq = wordscount_2.loc[index1,'Frequency']
                    term = term.split(" ")
                    word_freq.loc[word_freq.Word == term[0], 'Frequency'] = word_freq.loc[word_freq.Word == term[0], 'Frequency'] - freq
                    if term[0] != term[1]:
                        word_freq.loc[word_freq.Word == term[1], 'Frequency'] = word_freq.loc[word_freq.Word == term[1], 'Frequency'] - freq
                word_freq = word_freq.append(wordscount_2,ignore_index=True)

        word_freq = word_freq.sort_values(by = 'Frequency',ascending = False)
        word_freq['Word'] = word_freq['Word'].apply(lambda x: x.replace('_',' '))
        writer = str(folder + "/" + row[1] + "/" + "Column" + item + "_word_freq.xlsx")
        ensure_dir(writer)
        writer = pd.ExcelWriter(writer)
        word_freq.to_excel(writer,sheet_name="Term_frequency",index=None)
        writer.save()
        location = str(folder + "/" + row[1] + "/" + "Column" + item + "_wordcloud.png")
        word_freq = word_freq[['Word','Frequency']]
        word_freq = word_freq[word_freq['Frequency'] > 0]
        word_freq.reset_index(drop=True,inplace=True)
        data_for_topic.reset_index(drop=True,inplace=True)
        if str(masktype[n]).lower()=="n" or str(masktype[n])=='nan':
            savecloud(location=location,cloudtype=cloudtype[n],run=0,mask=demomask,prefer_horizontal=float(orientation[n]),font=fontpath)
        else:
            savecloud(location=location,cloudtype=cloudtype[n],run=1,mask=demomask,prefer_horizontal=float(orientation[n]),font=fontpath)
        location = str(folder + "/" + row[1] + "/" + "Column" + item + "_topics.xlsx")
        run_topic(data_for_topic,n_components,n_top_words,location)
        n+=1
        

