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
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mlp_model = joblib.load('mlp_model.pkl')
rfc_model = joblib.load('rfc_model.pkl')
nbc_model = joblib.load('nbc_model.pkl')


#check whether the message contains url
def has_url(text):
  url_regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.-]+[.][a-z]{2,})\S+)"
  return bool(re.findall(url_regex, text))


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
      for k in search(q,sleep_interval=5, num_results=10):
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
          extract2.append('NOT LEGITIMATE')
          print("NOT LEGITIMATE")
    return [extract,extract2]
  except:
    pass
  

#domain checking phase-2
def domainchec2(url):
  fp = urllib.request.urlopen(url)
  mybytes = fp.read()
  mystr = mybytes.decode("utf8")
  fp.close()
  new_str=mystr.split(" ")
  print()
  new_domain=[]
  for i in range(len(new_str)):
    myString = new_str[i]
    try:
      site=re.search("(?P<url>https?://[^\s]+)", myString).group("url")
      domain = extract_domain(site)
      new_domain.append(domain)
    except:
      pass
  return new_domain


#extract the domain
def extract_domain2(msg):
    site=re.search("(?P<url>https?://[^\s]+)", msg).group("url")
    extracted = tldextract.extract(site)
    domain = extracted.domain + "." + extracted.suffix
    return domain


#domain checking phase-2 main function
or_array=[]

def domain_checking_phase_2(extract_url, message):
  for i in range(len(extract_url)):
    try:
      check=extract_domain2(message)
      arr=domainchec2('https://www.'+extract_url[i])
      print(arr)
      for j in len(arr):
        if(arr[j]==check):
          print("LEGITIMATE")
          or_array.append('LEGITIMATE')
    except:
      pass
    finally:
      print("NOT LEGITIMATE")
      or_array.append('NOT LEGITIMATE')



#encode legitimate as 1 and non-legitimate as 0
or_array_encode=[]
def encode_list(arr):
  for m in arr:
    if(m == 'NOT LEGITIMATE'):
      or_array_encode.append(0)
    else:
      or_array_encode.append(1)
  return or_array_encode


#text feature extraction function
# def extract_features(text_message):
#     features_df = pd.DataFrame({'URL': [0], 'EMAIL': [0], 'PHONE': [0], 'special': [0], 'symbol': [0], 'misspelled': [0], 'common_words': [0]})

#     try:
#         site = re.search("(?P<url>https?://[^\s]+)", text_message).group("url")
#         if site:
#             features_df['URL'] = 1
#     except:
#         pass

#     def validate_email(email):
#         return bool(re.match(r"[^@]+@[^@]+\.[^@]+", email))

#     if validate_email(text_message):
#         features_df['EMAIL'] = 1

#     numbers = sum(c.isdigit() for c in text_message)
#     features_df['PHONE'] = 1 if numbers >= 10 else 0

#     special_characters = ['!', '$', '&', '#', '~']
#     features_df['special'] = sum(text_message.count(char) for char in special_characters)

#     symbol_characters = ['?', '%', '-', '/', '^', '%']
#     features_df['symbol'] = sum(text_message.count(char) for char in symbol_characters)

#     spell = SpellChecker()
#     wordlist = text_message.split()
#     amount_miss = len(list(spell.unknown(wordlist)))
#     features_df['misspelled'] = amount_miss

#     common_words = ["Award", "claim", "gift", "voucher", "blocked", "won", "prize", "winner", "activate", "please", "account", "card", "refund", "due", "congratulations", "cash", "urgent", "free", "happy", "join"]
#     features_df['common_words'] = sum(wordlist.count(word) for word in common_words)

#     return features_df



def extract_features(text_message):
    features_df = pd.DataFrame({'URL': [0], 'EMAIL': [0], 'PHONE': [0], 'special': [0], 'symbol': [0], 'misspelled': [0], 'common_words': [0]})
    extracted_values = []
    try:
        site = re.search("(?P<url>https?://[^\s]+)", text_message).group("url")
        if site:
            features_df['URL'] = 1
            extracted_values.append({'URL': {'count': 1, 'value': site}})
    except:
        pass

    def validate_email(email):
        return bool(re.match(r"[^@]+@[^@]+\.[^@]+", email))

    if validate_email(text_message):
        features_df['EMAIL'] = 1
        extracted_values.append({'EMAIL': {'count': 1, 'value': text_message}})

    numbers = sum(c.isdigit() for c in text_message)
    features_df['PHONE'] = 1 if numbers >= 10 else 0

    if features_df['PHONE'][0] == 1:
        phone_numbers = re.findall(r'\d{10,}', text_message)
        extracted_values.append({'PHONE': {'count': int(len(phone_numbers)), 'value': phone_numbers}})
    else:
        extracted_values.append({'PHONE': {'count': 0, 'value': []}})

    special_characters = ['!', '$', '&', '#', '~']
    features_df['special'] = sum(text_message.count(char) for char in special_characters)
    extracted_values.append({'special': {'count': int(features_df['special'][0]), 'value': special_characters}})

    symbol_characters = ['?', '%', '-', '/', '^', '%']
    features_df['symbol'] = sum(text_message.count(char) for char in symbol_characters)
    extracted_values.append({'symbol': {'count': int(features_df['symbol'][0]), 'value': symbol_characters}})

    spell = SpellChecker()
    wordlist = text_message.split()
    amount_miss = len(list(spell.unknown(wordlist)))
    misspelled_words = list(spell.unknown(wordlist))
    features_df['misspelled'] = amount_miss
    extracted_values.append({'misspelled': {'count': int(amount_miss), 'value': misspelled_words}})

    common_words = ["Award", "claim", "gift", "voucher", "blocked", "won", "prize", "winner", "activate", "please", "account", "card", "refund", "due", "congratulations", "cash", "urgent", "free", "happy", "join"]
    features_df['common_words'] = sum(wordlist.count(word) for word in common_words)
    common_words_values = [word for word in wordlist if word in common_words]
    extracted_values.append({'common_words': {'count': int(features_df['common_words'][0]), 'value': common_words_values}})

    return [features_df, extracted_values]



# message ="Dear customer, Bank of America is closing your bank account. Please confirm your PIN at to keep your account activated."
# features = extract_features(message)

#main function
# if has_url(message):
#   print('url exist')
#   result = domaincheck1(message)
#   print(result[0])
#   print(result[1])
#   check_url_legitimate = sum(encode_list(result[1])) > 1
#   if(check_url_legitimate):
#     print("check-1 Legitimate", check_url_legitimate)
#   else:
#     #domain checking phase-2
#     print("check-1 Not_Legitimate", check_url_legitimate)
#     domain_checking_phase_2(result[0], message)
#     check2_url_legitimate = sum(encode_list(or_array)) > 1
#     if(check2_url_legitimate):
#       print("check-2 Legitimate", check2_url_legitimate)
#     else:
#       print("check-2 Not_Legitimate", check2_url_legitimate)
#       #feature extraction model
#       features_df = extract_features(message)
#       if(model.predict(features_df)[0] == 1):
#         print("Legitimate check-3")
#       else:
#         print("Not_Legitimate check-3")
# else:
#   print('url not exist')
#   features_df = extract_features(message)
#   if(model.predict(features_df)[0] == 1):
#     print("Legitimate check-3")
#   else:
#     print("Not_Legitimate check-3")


# URL = features['URL'].iloc[0]
# EMAIL = features['EMAIL'].iloc[0]
# PHONE = features['PHONE'].iloc[0]
# special = features['special'].iloc[0]
# symbol = features['symbol'].iloc[0]
# misspelled = features['misspelled'].iloc[0]
# common_words = features['common_words'].iloc[0]

# print([[URL,EMAIL,PHONE,special,symbol,misspelled,common_words]])

# mlp_prediction = model.predict([[URL,EMAIL,PHONE,special,symbol,misspelled,common_words]])
# print("model output:", mlp_prediction) 



@app.get('/')
def index():
    return {'message': 'welcome to mlp classifier'}

@app.post("/multiplesms")
async def predict_multiple_sms(xml_file: UploadFile = File(...)):
    with open("received_xml.xml", "wb") as f:
        f.write(await xml_file.read())
    
    tree = ET.parse('received_xml.xml') 
    root = tree.getroot() 
    # print(root) 
    print(root[0].attrib["body"]) 
    sms = []
    prediction_list=[]

    for child in root:
        if 'body' in child.attrib:
            sms.append(child.attrib['body'])
    
    # Assuming you have a list of messages called 'messages'
    for message in sms:
        features = extract_features(message)[0]
        print(features)
        URL = features['URL'].iloc[0]
        EMAIL = features['EMAIL'].iloc[0]
        PHONE = features['PHONE'].iloc[0]
        special = features['special'].iloc[0]
        symbol = features['symbol'].iloc[0]
        misspelled = features['misspelled'].iloc[0]
        common_words = features['common_words'].iloc[0]
        pred = 0  # Initialize prediction

        predicted_model = mlp_model.predict([[URL,EMAIL,PHONE,special,symbol,misspelled,common_words]])
        
        if has_url(message):
            print('url exists')
            result = domaincheck1(message)
            print(result[0])
            print(result[1])
            check_url_legitimate = sum(encode_list(result[1])) > 1
            print("check URL1:",check_url_legitimate)
            if check_url_legitimate:
                pred = 1
                print("check-1 Legitimate", check_url_legitimate)
            else:
                print("check-1 Not_Legitimate", check_url_legitimate)
                domain_checking_phase_2(result[0], message)
                check2_url_legitimate = sum(encode_list(or_array)) > 1
                print("check URL2:", check2_url_legitimate)
                if check2_url_legitimate:
                    pred = 1
                    print("check-2 Legitimate", check2_url_legitimate)
                else:
                    print("check-2 Not_Legitimate", check2_url_legitimate)
                    if predicted_model[0] == 1:
                        pred = 1
                        print("Legitimate check-3")
                    else:
                        print("Not_Legitimate check-3")
        else:
            print('url does not exist')
            if predicted_model[0] == 1:
                pred = 1
                print("Legitimate check-3")
            else:
                print("Not_Legitimate check-3")

        print("prediction result", pred)

        if pred == 1:
            prediction = "Legitimate"
        else:
            prediction = "Not_Legitimate"
        
        # Append the prediction and message to a list
        prediction_list.append({"message":message, "prediction":prediction})
    
    return {
            "message": "XML file received and processed successfully",
            "smsList":sms,
            "predictedList":prediction_list
            }
   

@app.post('/sms')
def predict_sms_type(data:Schema):
    pred=0
    evaluation_metrics = [] 
    data = data.dict()
    message=data['message']
    model=data['model']
    print(model)
    print(message)

    features = extract_features(message)[0]
    print(features)
    URL = features['URL'].iloc[0]
    EMAIL = features['EMAIL'].iloc[0]
    PHONE = features['PHONE'].iloc[0]
    special = features['special'].iloc[0]
    symbol = features['symbol'].iloc[0]
    misspelled = features['misspelled'].iloc[0]
    common_words = features['common_words'].iloc[0]

    print([[URL,EMAIL,PHONE,special,symbol,misspelled,common_words]])


    feature_extract_values = extract_features(message)[1]
    print(feature_extract_values)



    if(model.__eq__("mlp")):
       predicted_model = mlp_model.predict([[URL,EMAIL,PHONE,special,symbol,misspelled,common_words]])
       print("predicted model", "mlp")
       evaluation_metrics.append([{"model":"MultiLayer perceptron"},{"description":"A multi-layer perceptron (MLP) is a type of artificial neural network consisting of multiple layers of neurons. The neurons in the MLP typically use nonlinear activation functions, allowing the network to learn complex patterns in data. "},{"accuracy":0.9456066945606695}, {"precision":0.9764373232799246}, {"recall":0.9628252788104089}, {"f1score":0.9695835283107159}])
    elif(model.__eq__("rfc")):
       predicted_model = rfc_model.predict([[URL,EMAIL,PHONE,special,symbol,misspelled,common_words]])
       print("predicted model", "rfc")
       evaluation_metrics.append([{"model":"Random Forest Classifier"},{"description":"Random Forest is a popular machine learning algorithm that belongs to the supervised learning technique. It can be used for both Classification and Regression problems in ML. It is based on the concept of ensemble learning, which is a process of combining multiple classifiers to solve a complex problem and to improve the performance of the model."},{"accuracy":0.9414225941422594}, {"precision":0.9644194756554307}, {"recall":0.9698681732580038}, {"f1score":0.9671361502347418}])
    elif(model.__eq__("nbc")):
       predicted_model = nbc_model.predict([[URL,EMAIL,PHONE,special,symbol,misspelled,common_words]])
       print("predicted model", "nbc")
       evaluation_metrics.append([{"model":"Naive Bayes Classifier"},{"description":"Naïve Bayes algorithm is a supervised learning algorithm, which is based on Bayes theorem and used for solving classification problems.Naïve Bayes algorithm is a supervised learning algorithm, which is based on Bayes theorem and used for solving classification problems.It is a probabilistic classifier, which means it predicts on the basis of the probability of an object."},{"accuracy":0.9229910714285714}, {"precision":0.9952021932830706}, {"recall":0.9172457359444094}, {"f1score":0.9546351084812623}])

      
    
    #main function
    if has_url(message):
      print('url exist')
      result = domaincheck1(message)
      print(result[0])
      print(result[1])
      check_url_legitimate = sum(encode_list(result[1])) > 1
      if(check_url_legitimate):
        pred=1
        print("check-1 Legitimate", check_url_legitimate)
      else:
        #domain checking phase-2
        print("check-1 Not_Legitimate", check_url_legitimate)
        domain_checking_phase_2(result[0], message)
        check2_url_legitimate = sum(encode_list(or_array)) > 1
        if(check2_url_legitimate):
          pred=1
          print("check-2 Legitimate", check2_url_legitimate)
        else:
          print("check-2 Not_Legitimate", check2_url_legitimate)
          #feature extraction model
          # features_df = extract_features(message)
          if(predicted_model[0] == 1):
            pred=1
            print("Legitimate check-3")
          else:
            print("Not_Legitimate check-3")
    else:
      print('url not exist')
      if(predicted_model[0] == 1):
        pred=1
        print("Legitimate check-3")
      else:
        print("Not_Legitimate check-3")

    print("prediction result", pred)


    if pred == 1:
      prediction="Legitimate"
    else:
      prediction="Not_Legitimate"
      
    return {
        'prediction': prediction,
        'extracted_features': feature_extract_values,
        'evalution_metrics': evaluation_metrics
    }


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)