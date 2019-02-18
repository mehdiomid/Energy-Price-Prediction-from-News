# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 20:47:09 2019

@author: Mehdi
"""
import pandas as pd
import os

def oilprice_sentiment(row):
    change = (row[0] - row[1]) / row[1]
    if change > 0.05:
        sentiment = "Higher"
    elif change < -0.05:
        sentiment = "Lower"
    else:
        sentiment = "NoChange"
         
    return sentiment 

articles_list = pd.read_csv("oilPrice_article_list1.csv")
oilprice = pd.read_csv("DCOILWTICO.csv")

oilprice["date"] = pd.to_datetime(oilprice["DATE"], format='%d/%m/%Y').map(lambda d: d.strftime('%Y-%m-%d'))
# convert the type of price from string to float
oilprice["oilprice"] = pd.to_numeric(oilprice["DCOILWTICO"], errors = 'coerce', downcast = 'float')

articles_list['date'] = pd.to_datetime(articles_list["date"]).map(lambda d: d.strftime('%Y-%m-%d')) 

# reading the text of all articles and create a data frame
path = "articles/"
files = os.listdir(path)
textList = []
for name in files:
    article_id = int(name[:-4])
    with open(path+name, "r", encoding="utf8") as f:
        text = f.read()
    textList.append((article_id, text))
textDataFrame = pd.DataFrame(textList, columns = ['article_id', 'text'])

articles = pd.merge(articles_list, textDataFrame, on = 'article_id', how = 'inner')  

#filling the NAs with previous value
oilprice.fillna(method='ffill', inplace=True)

oilprice.sort_values(by=['date'], inplace=True, ascending=True)
oilprice["price_9days"] = oilprice["oilprice"].shift(-9)
oilprice["price_21days"] = oilprice["oilprice"].shift(-21)
oilprice["price_50days"] = oilprice["oilprice"].shift(-50)

oilprice["sentiment_9days"] = oilprice[['price_9days', 'oilprice']].apply(oilprice_sentiment, axis=1)
oilprice["sentiment_21days"] = oilprice[['price_21days', 'oilprice']].apply(oilprice_sentiment, axis=1)
oilprice["sentiment_50days"] = oilprice[['price_50days', 'oilprice']].apply(oilprice_sentiment, axis=1)

articles_oilPrice = pd.merge(articles, oilprice, on = 'date', how = 'left') 

articles_oilPrice.to_csv("articles_oilPrice.csv", encoding = 'utf-8')

