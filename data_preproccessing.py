# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%%

import os
import xml.dom.minidom
import re


#%%
def get_data(dataFolder, testFile):
    xmlFile = os.path.join(dataFolder, testFile)
    ## creating document object model from the xml file
    DOMTree = xml.dom.minidom.parse(xmlFile)
    collection = DOMTree.documentElement
    ## getting author ID of the xml file 
    authorID = testFile.replace('.xml', '')
    ## getting language of the xml file 
    Language = collection.getAttribute("lang")
    ## getting data in the xml file
    Documents = collection.getElementsByTagName("document")
    Data = []
    for Document in Documents:
        if len(Document.childNodes) > 0:
            Data.append(Document.childNodes[0].data.encode('utf-8'))
        else:
            Data.append(' ')
    return Data, Language, authorID

#%%
file = open('/home/eileen/Documents/Language Technology Project/en/truth.txt', 'r+') 
Text = file.read()
file.close()
truthTexts = Text.split('\n')

genderList = []
authorTextList = []
languageList = []
authorIDList = []


for truth in truthTexts: 
    truthText = truth.split(':::')
    if len(truthText) == 3:
        xmlFile = truthText[0] + '.xml'
        Data, Language, authorID = get_data('/home/eileen/Documents/Language Technology Project/en', xmlFile)
        genderList.append(truthText[1])
        authorTextList.append(Data)
        languageList.append(Language)
        authorIDList.append(authorID)
    else: 
        pass
#%%        

       
# convert bytes to string
for i in range(0,len(authorTextList)):
    for j in range(0,len(authorTextList[i])):
        authorTextList[i][j] = authorTextList[i][j].decode()



# remove special characters, numbers, punctuations
for i in range(0,len(authorTextList)):
    for j in range(0,len(authorTextList[i])):
        authorTextList[i][j] = re.sub("http\S+", " ",authorTextList[i][j])
        authorTextList[i][j] = re.sub("[^a-zA-Z@]", " ",authorTextList[i][j]) # remove non a-z, A-Z or @
        authorTextList[i][j] = re.sub("@\S*", "",authorTextList[i][j]) # remove letters after @


#%%
from nltk.corpus import stopwords    
from nltk.tokenize import word_tokenize 

stop_words = set(stopwords.words('english'))    

for i in range(0,len(authorTextList)):
    for j in range(0,len(authorTextList[i])):
        authorTextList[i][j] = word_tokenize(authorTextList[i][j])
        authorTextList[i][j] = [w for w in authorTextList[i][j] if not w in stop_words] 


#%%
from nltk.stem import PorterStemmer 

stemmer = PorterStemmer()

for i in range(0,len(authorTextList)):
    for j in range(0,len(authorTextList[i])):
        authorTextList[i][j] = [stemmer.stem(w) for w in authorTextList[i][j]] 
        
        

#%%
    
import pickle

#save
with open('genderList', 'wb') as f:
    pickle.dump(genderList, f)

    
