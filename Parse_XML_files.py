#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 20:14:02 2019

@author: chenfish
"""

import os
from xml.dom.minidom import parse
import xml.dom.minidom




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

file = open('/Users/chenfish/Downloads/training/en/truth.txt', 'r+') 
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
        Data, Language, authorID = get_data('/Users/chenfish/Downloads/training/en/', xmlFile)
        genderList.append(truthText[1])
        authorTextList.append(Data)
        languageList.append(Language)
        authorIDList.append(authorID)
    else: 
        pass
        
    #print (truthText)


