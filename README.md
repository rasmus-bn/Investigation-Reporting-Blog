# UFO-Blog

#### *  Abstract*

## Introduction

The target audience for this post is software development students whom might not have dipped their toes in machine learning. I will therefore refrase some technical terms in order to ease the readability of the user. Despite me being a bit liberal 

## Initiating problem statement

In a world with a continously rising amout of data it is essential to be able to automate the processing of the data. Machine learning or "AI" is a great tool for analyzing large amounts of data. However the algorithms are based on math and can not be directly fed with images or text data. The algorithms only understands numbers so in order to use machine learning to process texts or images the data needs to be preprocessed.



(Converting text into feature vectors.)




## Narrowing (What to choose?)

In my machine learning [exam project](https://github.com/rasmus-bn/MLExamProject 'ML Exam Project') I performed sentiment analysis on workplace reviews created by employees at some of the worlds largest tech companies. The dataset contained text data written by the employee alongside with a rating from 1-5. We had to classify whether the rating was negative (1-3) or positive (4-5) based on the text written by the employee. We therefore needed to transform the data into numbers for our machine learning algorithm to understand.

## Problem statement

*Which method should I use to transform text data into a format that a machine learning alogrithm can understand?*

## First some technical terms

Before diving into it I want to explain some terms that I will be using.

* **Feature**: A feature is an input variable that the machine learning algorithm uses to calculate its output. Say you want a machine learning model be able to identify the age of a person based on the persons height and weight. In this case there are 2 features height and weight.

## The different methods

We mainly used [scikit-learn](https://scikit-learn.org/stable/index.html# "scikit-learn home page") so I want to go through some of the methods that scikit-learn has available to perform this task.

### Label encoding

Label encoding means identifying all unique values of a specific feature in a dataset and assigning an integer number to that value. Ex. I have a feature called color. In the dataset I the only unique values are 'green', 'blue' and 'red'. The [LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html "scikit-learn on OneHotEncoder") will then assign a number from 0-3 to each of the uniqe values: 0 = red, 1 = blue, 2 = green.

The downside of this approach is that it implies that there are that the colors is a sort of range where red is the smallest value and green is the greatest. The machine lerning algorithm only gets the numbers so it are not able to destinguish between the colors.

### One-hot encoding

The [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html "scikit-learn on OneHotEncoder")

### Encoding based on occurence count

The limit of the 
There are multiple ways to do this, one of which is the CountVectorizer. The CountVectorizer creates a matrix of numbers counting the occurences of corresponding words. Below is a simple example of the CountVectorizer in use.

![alt text](https://raw.githubusercontent.com/rasmus-bn/Investigation-Reporting-Blog/master/images/CountVectorizer%20example.png "See 'CountVectorizer example.ipynb' in this repo")



### Count vectorization

## Problem statement

Why remove stop words when using tfidf?
Using tfidf with stopwords?
Tfidf vs countvectoriser?

How tfidf works and why we chose to use it?
We chose tfidf but is it better?





RabbitMQ - Guide

ML feature extraction on text


TFIDF vecotorizer vs Count vecotrizer
https://www.kaggle.com/c/quora-question-pairs/discussion/31273



removing stopwords when using tfidf