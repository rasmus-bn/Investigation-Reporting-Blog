# Text encoding for use in machine learning

*Posted by Rasmus Nordbjærg at https://github.com/rasmus-bn/Investigation-Reporting-Blog on December 2019*

***Artificial Intelligence algorithms doesn't understand raw text data. However the world is drowning in text data and Artificial Intelligence is our best tool for processing and analyzing large data quantities. Fortunately text data can be encoded into a format that are understandable by Artificial Intelligence. Encoding text data allows you to ride the wave of Artificial Intelligence in the expanding ocean of collected text data.***


## Introduction

In a world with a continously rising amout of data it is essential to be able to automate the processing of the data. Machine learning or artificial intelligence is a great tool for analyzing large amounts of data. However the algorithms are based on math and can not be directly fed with images or text data. The algorithms only understands numbers so in order to use machine learning to process texts or images the data needs to be preprocessed.

Machine learning models work by calculating a result *y* from the input variable *X*. However a model can have multiple input and output variables. Input variables are called **features** and the collection of input variables for a machine learning model is called a **feature vector**.

An example case of a machine learning model could be this: *The model should predict how much more time a pizza should stay in the oven before it is done. We know how long the pizza has already been in the oven and we know the oven temperature so these would be the features (input variables)*:
* X<sub>1</sub> = time passed
* X<sub>2</sub> = oven temperature

*We want to predict how much time is left before we can the pizza out:*
* y = time left

The important thing to take away is that the model only understand *X*s and the *y*s and these variables can only be given numerical values. There exists multiple methods for encoding text data into numerical features and I want to compare some of these. In the end I want to answer the below question:

*How can I encode text data so that a machine learning algorithm can understand the data?*

## First some technical terms

Before diving into it I want to explain some words that I will be using:

* **Feature**: A feature is an input variable that the machine learning algorithm uses to calculate its output. Say you want a machine learning model be able to identify the age of a person based on the persons height and weight. In this case there are 2 features height and weight.

* **Corpus**: A collection of texts.

* **Document**: A specific text in a corpus.

Another note is that I will only explain different methods for encoding data. I will therefore not dive into word stemming, removal of stopwords or other ways of preparing the text data.

## Bag of Words

In my machine learning exam project<sup>**</sup> I had to predict whether a review was positive or negative based on the written text of the reviewer. This sort of analysis is called sentiment anelysis and I will base my explanations on that kind of use case. So the input is a text that needs to be encoded into features and the ouput is a binary value: positive or negative.

Bag of Words means having one feature for every unique word in the dataset. If a dataset only contains 3 unique words the input for the machine learning model would consist of three features. In order to represent a text the features would be assigned numerical values. A scoring method is used to calculate what the numerical value should be. I will got through the three scoring methods that we considered in my machine learning project.

Bag of Words only tracks what words are mentioned in the text and not in which order the words occured. This means that the original text can not be recovered from the encoded data. However in the use case of sentiment analysis the purpose is to predict whether the inputted text is negative or positive so we don't need the ability to recover the text from the encoded representation.

***I will only explain different methods for encoding data. I will therefore not dive into word stemming, removal of stopwords or other ways of preparing the text data. I will also leave out exlpaining n-grams for the sake of simplicity and readability.***

#### Binary scoring method

With binary scoring each method can have either 0 or 1. If a feature is given the value of 1 it means that the corresponding word exists the text and 0 means that it doesn't. Below is an image demonstrating this in python code.

![alt text](https://raw.githubusercontent.com/rasmus-bn/Investigation-Reporting-Blog/master/images/One-Hot%20example.png "See 'Encoding examples.ipynb' in the repo")
*https://raw.githubusercontent.com/rasmus-bn/Investigation-Reporting-Blog/master/images/One-Hot%20example.png - A screenshot from 'Encoding examples.ipynb' in the blog repo*

The limit of this method is that it does not take into account the number of occurences of the words. These 2 sentences would equal eachother if encoded with the binary scoring method: "It really was a good movie", "It was a really really good movie".

#### Count based scoring

Like one-hot encoding the CountVectorizer<sup>5</sup> creates a feature for each unique value. However instead of just telling whether the word is used in a sentece the CountVectorizer counts the number of occurences of the word. Below is a simple example of the CountVectorizer in use:

![alt text](https://raw.githubusercontent.com/rasmus-bn/Investigation-Reporting-Blog/master/images/Count-based%20example.png "See 'Encoding examples.ipynb' in the repo")

Using this method means that the algorithm would if a word occurs more than once in a text.

#### TF-IDF


![alt text](https://raw.githubusercontent.com/rasmus-bn/Investigation-Reporting-Blog/master/images/TF-IDF%20example.png "See 'Encoding examples.ipynb' in the repo")

The CountVectorizer seemed like it had all the functionality that we needed however that was until we read about the TfidfVectorizer<sup>6</sup>. TF-IDF stands for Term Frequency - Inverse Document Frequency<sup>7</sup>. Below is an example of what the TF-IDF does:

* The word 'red' occurs many times in the entire datase. In one document 'red' occurs 3 times and gets the weight of '0.21'.

* The word 'green' only occurs a few places in the dataset. However in this document 'green' occurs 3 times resulting in the weight of '0.75'

As shown in the example the more common words used in the dataset will be weighted as less important than the more rare words. Similar to the CountVectorizer the TfidfVectorizer expresses the number of occurences of each word. However it will evaluate less common words to be more important than frequent words. This means that if many reviews both positives and negatives would mention the word 'workplace' then 'workplace' does not bring any specific meaning to the review. The TfidfVectorizer takes care of down prioritising those kind of words and scikit-learn reccommends<sup>8</sup> TfdifVectorizer over CountVecotizer when working with text data. This was also the method we went with.

## Conclusion
TF-IDF is a great method for encoding text data to feed into a machine learning algorithm. It provides a score for each word, based on how many times the word occurs in a text and how rare that word is. Because of these functionalities I believe TF-IDF was the best of the proposed methods for encoding text data in our machine learning algorithm.

## Follow-up reflection

Encoding the text data is only a small step of building the classification model. We had multiple data preprocessing steps that we went through such as word stemming, stopwords removal, removal of company names (to not be biased if a company name is mentioned) and other text cleanup. We also engineered some extra features such as word count, stopword count & frequency, text length in characters.

We used TF-IDF in our machine learning exam project and we managed to get a prediction accuracy of arround 80%. However this result was not credited alone to TF-IDF as many other factors were in place.

## References

* My machine learning exam project (state of master branch 10/12-19): https://github.com/rasmus-bn/MLExamProject

* scikit-learn home page (last read on 10/12-19): https://scikit-learn.org/stable/index.html#

* scikit-learn's LabelEncoder (last read on 10/12-19): https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html

* scikit-learn's OneHotEncoder (last read on 10/12-19): https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html

* scikit-learn's CountVectorizer (last read on 10/12-19): https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

* scikit-learn's TfidfVectorizer (last read on 10/12-19): https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

* Wikipedia's explanation of Term Frequency - Inverse Document Frequency (last read on 10/12-19): https://en.wikipedia.org/wiki/Tf%E2%80%93idf

* scikit-learn's guide on working with text data recommending TFIDFVectorizer over CountVectorizer (last read on 10/12-19): https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

* Wikipedia on Features in machine learning (last read on 10/12-19): https://en.wikipedia.org/wiki/Feature_(machine_learning)

* Wikipedia on one-hot https://en.wikipedia.org/wiki/One-hot

* Bag of words - Jason Brownlee 09/10-19 (last read on 10/12-19) https://machinelearningmastery.com/gentle-introduction-bag-words-model/

* Wikipedia on Bag of Words (last read on 10/12-19): https://en.wikipedia.org/wiki/Bag-of-words_model
