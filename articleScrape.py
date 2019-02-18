# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 20:40:34 2019

@author: Mehdi
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import io

def articleFetch(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    text_elements = soup.find('div', {"id":"article-content"}).findAll('p')
    text = " ".join([text_element.get_text() for text_element in text_elements])
    return text

def articleStore(text, article_id):

    fname = 'articles/'+str(article_id)+'.txt'
    with io.open(fname, "w", encoding="utf-8") as f:
        f.write(text)
    return "Downloaded"

def articleDownload(row):
    text = articleFetch(row['link'])
    _= articleStore(text, row['article_id'])
    return "Stored"

titles = []
links = []
dates = []
summaries = []
image_links = []
for page_number in range(1, 201):
    page = requests.get("https://oilprice.com/archive/Page-"+str(page_number)+".html")
    soup = BeautifulSoup(page.content, 'html.parser')
    
    for row in soup.find_all('div', class_='categoryArticle'):
        titles.append(row.find_all('h2', class_='categoryArticle__title')[0].get_text())
        links.append(row.find_all('a')[0]['href'])
        dates.append(row.find_all('p', class_="categoryArticle__meta")[0].get_text())
        summaries.append(row.find_all('p', class_= "categoryArticle__excerpt")[0].get_text())
        image_links.append(row.find_all('img')[0]['src'])

articles = pd.DataFrame({
        "title": titles, 
        "link": links,
        "date": dates,
        "summary": summaries,
        "image_link": image_links
    })
articles['article_id'] = articles.index

articles['date'] = pd.to_datetime(articles["date"]).map(lambda d: d.strftime('%d/%m/%Y')) 

#pd.set_option('display.max_colwidth', -1)
articles.to_csv("oilPrice_article_list1.csv", encoding = 'utf-8')
    
articles2 = pd.read_csv("oilPrice_article_list1.csv")#, encoding = "ISO-8859-1")
articles2.apply(articleDownload, axis = 1)

