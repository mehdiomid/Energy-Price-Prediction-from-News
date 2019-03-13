# Energy-Price-Prediction-from-News
This is a repo for energy price prediction from news using advanced NLP and Recurrent Neural Networks LSTM Deep Learning

This is a project to show that the energy price can be predicted using news articles. In this phase, it is aimed to predict the short-term price movements of crude oil. Natural language processing and machine learning techniques are used to explore solutions to this problem. 

## Data collection
The news articles were scraped from oilprice.com as of August 2015, using the BeautifulSoup python package. 

## Label the training and test set
Given the price of crude oil over time, for each time interval the price movements were classified as "up," "down," or "unchanged”. The price movement was considered significant enough to mark it “up” or “down”, if the price change was at least 5%. Each article in the training set of news articles was then labeled "up," "down," or "unchanged" according to the movement of the crude oil price in a time interval associated with the publication date of the article. 

## Text preprocessing and feature engineering
From each text, the metadata such as the date and time a document was published and the categories of disclosures made were extracted, while tables and charts were discarded. All texts were preprocessed by removing stopwords, punctuation, and numbers, lemmatizing words and converting them to lower case. All documents were tokenized and converted to sequences and then padded with zeros to a uniform length.
The text have been processed to create unigram models and then were used to make an embedding layer for the LSTM model. A pretrained embedding model, named GloVe, which was the Stanford NLP embedding set was used. The pretrained embedding was trained on Wikipedia and have vectors for many of the industry-specific words found in the articles. 

## Train a deep learning model and evaluate it on AWS EC2
A deep learning text classifier was then trained to predict which movement class an article belongs to. Given a test article, the trained classifier potentially predicts the price movement of the crude oil price. 
A deep learning model was constructed using Keras with a Tensorflow backend, consisting of an input layers for text documents, an embedding layer with the pretrained GloVe vectors, and three LSTM layers, followed by a dense layer with softmax as the activation function. 
The python code to train the model was executed on an AWS EC2 instance using deep learning ubuntu Environment. To evaluate the model, the dataset was randomly shuffled and split into 80% train and 20% test data. The model achieved an accuracy of 0.85 on the training set and 0.83 on the test set.

## Predictions
The trained model is used to to predict whether the price of crude oil will go up, down, or stay approximately unchanged within a time frame after disclosing the new articles. For this purpose, a cron job is scheduled to scrape the news articles published yesterday, process the text and use it to make a new prediction. The result is being stored on AWS s3 and then displayed on the website hosted on heroku.
