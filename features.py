import uvicorn
from fastapi import FastAPI
import joblib
from textblob import TextBlob
import pandas as pd
import nltk
from googlesearch import search
import tldextract
import urllib.request
import re
import pandas as pd
from spellchecker import SpellChecker
from schema import Schema
import ssl
from fastapi import FastAPI, File, UploadFile
import xml.etree.ElementTree as ET 

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('brown')
nltk.download('punkt')

#extract the domain name from url
def extract_domain(url):
    extracted = tldextract.extract(url)
    domain = extracted.domain + "." + extracted.suffix
    return domain


#domain checking phase -1
def domaincheck1(msg):
  try:
    extract=[]
    extract2=[]
    site=re.search("(?P<url>https?://[^\s]+)", msg).group("url")
    domain = extract_domain(site)
    text = TextBlob(msg)
    save2=text.noun_phrases
    print(save2)
    for j in range(len(save2)):
      q=save2[j]+" "+ domain
      links = []
      for k in search(q):
        links.append(k)
      print(links)
      for l in range(len(links)):
        extract_link=extract_domain(links[l])
        if(extract_link==domain):
          print(domain, " LEGITIMATE")
          extract2.append('LEGITIMATE')
          break
        else:
          extract.append(extract_link)
          extract2.append('NOTLEGITIMATE')
          print("NOTLEGITIMATE")
    return [extract,extract2]
  except:
    pass
  
  

print(domaincheck1("Lodge complaint against any bank, NBFC or payment system participant on https://cms.rbi.org.in under RB-Integrated Ombudsman Scheme. Call 14440 for more -RBI"))