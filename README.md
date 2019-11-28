# Text encoding for use in machine learning

## Introduction

In a world with a continously rising amout of data it is essential to be able to automate the processing of the data. Machine learning or "AI" is a great tool for analyzing large amounts of data. However the algorithms are based on math and can not be directly fed with images or text data. The algorithms only understands numbers so in order to use machine learning to process texts or images the data needs to be preprocessed.

In my machine learning 
[exam project](https://github.com/rasmus-bn/MLExamProject 'ML Exam Project')
I performed sentiment analysis on workplace reviews created by employees at some of the worlds largest tech companies. The dataset contained text data written by the employee alongside with a rating from 1-5. We had to classify whether the rating was negative (1-3) or positive (4-5) based on the text written by the employee. We therefore needed to transform the data into numbers for our machine learning algorithm to understand.

*Which method should I use to encode text data into a format that a machine learning alogrithm can understand?*

## First some technical terms

Before diving into it I want to explain some words that I will be using:

* **Feature**: A feature is an input variable that the machine learning algorithm uses to calculate its output. Say you want a machine learning model be able to identify the age of a person based on the persons height and weight. In this case there are 2 features height and weight.

* **Corpus**: A collection of texts.

* **Document**: A specific text in a corpus.

Another note is that I will only explain different methods for encoding data. I will therefore not dive into word stemming, removal of stopwords or other ways of preparing the text data.

## Methods for encoding text into features

We mainly used 
[scikit-learn](https://scikit-learn.org/stable/index.html# "scikit-learn home page") 
so I want to go through some of the methods that scikit-learn has available to perform this task.

#### Label encoding

Label encoding means identifying all unique values of a specific feature in a dataset and assigning an integer number to that value. Ex. I have a feature called color. In the dataset I the only unique values are 'green', 'blue' and 'red'. The 
[LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html "scikit-learn's OneHotEncoder") 
will then assign a number from 0-3 to each of the uniqe values: 0 = red, 1 = blue, 2 = green.

The downside of this approach is that it implies that that the colors is a sort of range where red is the smallest value and green is the greatest. The machine lerning algorithm only gets the numbers so it are not able to destinguish between the colors.

Another problem with this encoding method is that the input can only be a single word. However we were dealing with entire sentences so this method would not allow us to pass in the data.

#### One-hot encoding

With one-hot encoding every unique value is turned into a binary feature. This means that every word in the dataset will be represented by a feature that can either be 0 (word not used) or 1 (word was used). This way all words in the text can be encoded which was what we needed.

See below example:

|                         | I | am | blue | Dan | is | red | and |
|-------------------------|---|----|------|-----|----|-----|-----|
| I am blue               | 1 | 1  | 1    | 0   | 0  | 0   | 0   |
| Dan is red              | 0 | 0  | 0    | 1   | 1  | 1   | 0   |
| I am red and Dan is red | 1 | 1  | 0    | 1   | 1  | 1   | 1   |

The increase of the number of features might pose some other issues but it gives the ability to encode all words in a text which was what we needed. Label encoding did not provide this functionality. Besides the [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html "scikit-learn's OneHotEncoder")
also allows limiting the number of features produced.

The limit of this method is that it does not take into account the number of occurences of the words. These 2 sentences would equal eachother after one-hot encoding: "It was a bad movie", "It was a bad bad bad bad bad movie".

#### Encoding based on occurence count

Like one-hot encoding the
[CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html "scikit-learn's CountVectorizer")
creates a feature for each unique value. However instead of just telling whether the word is used in a sentece the CountVectorizer counts the number of occurences of the word. Below is a simple example of the CountVectorizer in use:

![alt text](https://raw.githubusercontent.com/rasmus-bn/Investigation-Reporting-Blog/master/images/CountVectorizer%20example.png "See 'CountVectorizer example.ipynb' in this repo")

Using this method means that the algorithm would if a word occurs more than once in a text.

#### TF-IDF

The CountVectorizer seemed like it had all the functionality that we needed however that was until we read about the [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html "scikit-learn's TfidfVectorizer"). TF-IDF stands for [Term Frequency - Inverse Document Frequency](https://en.wikipedia.org/wiki/Tf%E2%80%93idf "Wikipedia's explanation of TF-IDF"). Below is an example of what the TF-IDF does:

* The word 'red' occurs many times in the entire datase. In one document 'red' occurs 3 times and gets the weight of '0.21'.

* The word 'green' only occurs a few places in the dataset. However in this document 'green' occurs 3 times resulting in the weight of '0.75'

As shown in the example the more common words used in the dataset will be weighted as less important than the more rare words. Similar to the CountVectorizer the TfidfVectorizer expresses the number of occurences of each word. However it will evaluate less common words to be more important than frequent words. This means that if many reviews both positives and negatives would mention the word 'workplace' then 'workplace' does not bring any specific meaning to the review. The TfidfVectorizer takes care of down prioritising those kind of words and scikit-learn [reccommends](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html "'Working With Text Data' guide") TfdifVectorizer over CountVecotizer when working with text data. This was also the method we went with.

## Conclusion
TF-IDF is a great method for encoding text data to feed into a machine learning algorithm. It provides a score for each word, based on how many times the word occurs in a text and how rare that word is. Because of these functionalities I believe TF-IDF was the best of the proposed methods for encoding text data in our machine learning algorithm.

## Follow-up reflection

Encoding the text data is only a small step of building the classification model. We had multiple data preprocessing steps that we went through such as word stemming, stopwords removal, removal of company names (to not be biased if a company name is mentioned) and other text cleanup. We also engineered some extra features such as word count, stopword count & frequency, text length in characters.

We used TF-IDF in our machine learning exam project and we managed to get a prediction accuracy of arround 80%. However this result was not credited alone to TF-IDF as many other factors were in place.