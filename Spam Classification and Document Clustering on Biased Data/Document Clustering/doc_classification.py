#In trainning mode run the program as follows - 
#    python doc_classification.py train train Ulearning.txt 1.0
#   It takes around 3 minutes to learn the model in case of supervised learning, while it takes around 7-8 minutes in case of semi supervised learning.
#In testing mode run the program as follows - 
#    python doc_classification.py test test Ulearning.txt 1.0
#    It takes around 10 minutes to read through all the files give and the ouput
'''
References:
1) HTML Tag Remover : http://stackoverflow.com/questions/9662346/python-code-to-remove-html-tags-from-a-string
2) Writing a Dictionary to File : http://stackoverflow.com/questions/899103/writing-a-list-to-a-file-with-python
3) Christopher Manning's PPT: https://web.stanford.edu/class/cs124/lec/naivebayes.pdf
4) Stop words - http://sebastianraschka.com/Articles/2014_naive_bayes_1.html
These are the helper files that will help the program to retrieve, read, and clean the data from files.
'''
# Training mode - 
# The program loops over all the topics present in directory one by one - 
    #In every topic, the words present in each file are added to a list. So for every topic, the program returns list of words which are present in trainning set.
    #AFter performing some data cleaning n word list, the program calculated naiive bayes model
    #The probability of word given a topic is  calculated as frequency of word in that topic over total number of words in that topic.
    #The probability model is saved in likelihood dictionary
#Testing mode,
#The program uses max apriori method to classify the document into a topic.

import os
import pandas as pd
import operator
from collections import OrderedDict
from collections import Counter
import time
import sys
import re
import math
import pickle
import random
arguments = sys.argv
output_file = open(str(os.getcwd())+'/'+'distinctive words.txt','w')
output_file.close()
likelihood = {}
model_fileName = arguments[3]
dataset_dir = arguments[2]
fraction = float(arguments[4])
org_path = str(os.getcwd()) #saving the initial path at which program is residing


#readFiles returns the words present in document for the topic provided in path variable. It return a list of words which are present in that topic
def readFiles(path):
    os.chdir(path)
    file_content = []
    file_name_list = os.listdir(path)
    print "no of iles - ",len(file_name_list)
    for filename in file_name_list:
        f = open(filename, 'r')
        linetext=f.readlines()
        words=[]
        for line in linetext:
            line = line.rstrip('\n').rstrip('\t')
            words.append(line.split(' '))
        file_content.append([everyWord for everyLine in words for everyWord in everyLine])
        f.close()
    return [item for sublist in file_content for item in sublist]



#readTestFile returns list of words present in a file.
def readTestFile(filename):
    f = open(filename, 'r')
    linetext=f.readlines()
    words=[]
    for line in linetext:
        line = line.rstrip('\n').rstrip('\t')
        words.append(line.split(' '))
    #file_content.append([everyWord for everyLine in words for everyWord in everyLine])
    f.close()
    return [item for w in words for item in w]

#cleanData removes numeric values, symbols, and stop words from the list provided to the method
def cleandata(file_folder):
    # Cleaner_1 removes HTML Tags
    clean_files = []
    for f in file_folder:
        cleaner_1 = re.compile('<[^<]+?>')
        clean_text_1 = re.sub(cleaner_1, '', f)
    # Cleaner_2 removes special characters
        cleaner_2 = re.compile('[!&@:;]')
        clean_text_2 = re.sub(cleaner_2, '', clean_text_1)
    # Cleaner_3 removes all types of brackets
    #cleaner_3 = re.compile('[<[{()}]>]')
        clean_text_3 = re.sub('[[]]', '', clean_text_2)
        clean_text_4 = re.sub('[()]', '', clean_text_3)
        clean_text_5 = re.sub('[<>]', '', clean_text_4)
        clean_text_6 = re.sub('[{}]', '', clean_text_5)
        clean_text_7 = re.sub('[_-]', '', clean_text_6)
        clean_text_8 = re.sub('["]', '', clean_text_7)
        clean_text_9 = re.sub("[']", '', clean_text_8)
        clean_text_10 = re.sub('[.,]', '', clean_text_9)
        clean_text_11 = re.sub('[*=]', '', clean_text_10)
        clean_text_12 = re.sub('[/]', '', clean_text_11)
        clean_text_13 = re.sub('#', '', clean_text_12)
        clean_text_14 = re.sub('%', '', clean_text_13)
        clean_text_15 = re.sub('[?]', '', clean_text_14)
        clean_text_16 = re.sub('[~]', '', clean_text_15)
        clean_text_17 = re.sub('[|]', '', clean_text_16)
        clean_text_18 = re.sub('[\t]', ' ', clean_text_17)
        clean_text_19 = re.sub('[^a-zA-Z]','',clean_text_18)#letters only, no numeric value
        clean_files.append(clean_text_19)
    words =  [value for value in clean_files if value != '']
    stopwords = [u'i', u'me', u'my', u'myself', u'we', u'our', u'ours', u'ourselves', u'you', u'your', u'yours', u'yourself', u'yourselves', u'he', u'him', u'his', u'himself', u'she', u'her', u'hers', u'herself', u'it', u'its', u'itself', u'they', u'them', u'their', u'theirs', u'themselves', u'what', u'which', u'who', u'whom', u'this', u'that', u'these', u'those', u'am', u'is', u'are', u'was', u'were', u'be', u'been', u'being', u'have', u'has', u'had', u'having', u'do', u'does', u'did', u'doing', u'a', u'an', u'the', u'and', u'but', u'if', u'or', u'because', u'as', u'until', u'while', u'of', u'at', u'by', u'for', u'with', u'about', u'against', u'between', u'into', u'through', u'during', u'before', u'after', u'above', u'below', u'to', u'from', u'up', u'down', u'in', u'out', u'on', u'off', u'over', u'under', u'again', u'further', u'then', u'once', u'here', u'there', u'when', u'where', u'why', u'how', u'all', u'any', u'both', u'each', u'few', u'more', u'most', u'other', u'some', u'such', u'no', u'nor', u'not', u'only', u'own', u'same', u'so', u'than', u'too', u'very', u's', u't', u'can', u'will', u'just', u'don', u'should', u'now', u'd', u'll', u'm', u'o', u're', u've', u'y', u'ain', u'aren', u'couldn', u'didn', u'doesn', u'hadn', u'hasn', u'haven', u'isn', u'ma', u'mightn', u'mustn', u'needn', u'shan', u'shouldn', u'wasn', u'weren', u'won', u'wouldn']
    return [w for w in words if not w in stopwords]


def convert_to_lower(data):
    lower = []
    for d in data:
        lower.append(d.lower())
    return lower

#In semi supervised learning update_likelihood updates the model probability after each sampling iteration
def update_likelihood(each_topic ,word_prob):
    if each_topic in likelihood:
        existing_word_prob = likelihood[each_topic]
        for k,v in word_prob.iteritems():
            if k in existing_word_prob:
                existing_word_prob[k] = (existing_word_prob[k]+v)/2.0 # taking average of the two probabilities
            else: # new wprd
                existing_word_prob[k] = v 
        likelihood[each_topic] = existing_word_prob
    else:
        likelihood[each_topic] = word_prob

# It returns topic of the file by deducing from naive bayes model.
def find_topic(eachFile):
    words = list(OrderedDict.fromkeys(cleandata(convert_to_lower(readTestFile(eachFile)))))#eliminating duplicate words
    posterior =  dict((k,sum(likelihood[k][everyWord] if everyWord in likelihood[k].keys() else math.log(0.0000001) for everyWord in words)) for k in likelihood.keys())
    deducted_topic = max(posterior, key=posterior.get)
    return deducted_topic


def find_topic_Unsupervised(words):
    words = list(OrderedDict.fromkeys(words))#eliminating duplicate words
    posterior =  dict((key,sum(value[everyWord] if everyWord in value else math.log(0.0000001) for everyWord in words)) for key,value in likelihood.iteritems())
    #posterior =  dict((k,sum(likelihood[k][everyWord] if everyWord in likelihood[k].keys() else math.log(0.0000001) for everyWord in words)) for k in likelihood.keys())
    deducted_topic = max(posterior, key=posterior.get)
    return deducted_topic

def readFilesByTopic(topic,path):
    os.chdir(path+'/'+topic)
    filenames = os.listdir(path+'/'+topic)
    files=dict((eachFile,list(OrderedDict.fromkeys(cleandata(convert_to_lower(readTestFile(eachFile))))))for eachFile in filenames)
    return files

#for trainning the model
if arguments[1]=='train':
    topics=[]
    words=[]
    path = (str(os.getcwd())+'/'+'train') # changind directory to train
    os.chdir(path)
    topics = os.listdir(path) # topics list contains name of all the topics which are present in train directory
    mxW = 0
    if  fraction == 1.0:
        # calculate likelihood  probability of word given topic
        #for a given topic, count the frequency of each words
        #probability of a word in a given topic is frequency of word over total number of words appeared in that topic
        print "learning started ..."
        s = time.time()
        for each_topic in topics:
            print "learning %s topic"%(each_topic)
            s1=time.time()
            word_prob ={}
            words = cleandata(convert_to_lower(readFiles(path+'/'+each_topic)))
            if len(words)>mxW:
                mxW = len(words)
            word_prob = dict(Counter(words))
            word_prob = dict((key,math.log(float(value)/len(words),2))for key,value in word_prob.items())
            likelihood[each_topic] = word_prob
            print "learning %s got completed in"%(each_topic),(time.time()-s1),"secs"
        print "learning all topics completed in ",(time.time()-s),"secs"
    elif fraction>0.0 and fraction<1.0:
        readDoc=[]
        print "learning started ..."
        print "Kindly wait for around 5-7 minutes for semi supervised learning to get complete"
        sU = time.time()
        all_files=dict((eachTopic,readFilesByTopic(eachTopic,path))for eachTopic in topics)
        os.chdir(org_path)
        unreadDoc=[]
        doc_topic = dict((each,[])for each in topics)
        for i in range(0,5):
            s=time.time()
            for each_topic,topicFiles in all_files.iteritems():
                for eachFile,words in topicFiles.iteritems():
                    if float(random.randint(0,10))/10 <= fraction:# read the file and calculate the probability
                        readDoc.append(eachFile)
                        doc_topic[each_topic].append(eachFile)
                        word_prob ={}
                        word_prob = dict(Counter(words))
                        word_prob = dict((key,math.log(float(value)/len(words),2))for key,value in word_prob.iteritems())
                        update_likelihood(each_topic, word_prob)
                    else:# can't read the topic for document hence saved in unread dicument 
                        unreadDoc.append(eachFile)   
            #Test on all those remaining unread documents and find out the its topic
            for eachUnreadFIle in unreadDoc:
                words = [topicFiles[eachUnreadFIle] for topic,topicFiles in all_files.iteritems() if eachUnreadFIle in topicFiles][0]
                deducted_topic = find_topic_Unsupervised(words)
                if eachUnreadFIle not in doc_topic[deducted_topic]:
                    [doc_topic[dt].remove(eachUnreadFIle) for dt,flist in doc_topic.iteritems() if eachUnreadFIle in flist]
                    doc_topic[deducted_topic].append(eachUnreadFIle)
        print "learaning got completed in ",(time.time()-sU),"secs"
    elif fraction==0.0:
        print "learning started ..."
        print "Kindly wait for 5-7 minutes for unsupervised learning to get complete"
        sU1 = time.time()
        all_files=dict((eachTopic,readFilesByTopic(eachTopic,path))for eachTopic in topics)
        os.chdir(org_path)
        doc_topic = dict((each,[])for each in topics)
        unreadDoc=[]
        for i in range(0,5):
            s=time.time()
            for each_topic,topicFiles in all_files.iteritems():
                for eachFile,words in topicFiles.iteritems():
                    unreadDoc.append(eachFile)
                    doc_topic[each_topic].append(eachFile)
                    word_prob ={}
                    #generataeProb(words)
                    word_prob = dict(Counter(words))
                    word_prob = dict((key,random.randint(-20000,0)/1000.0)for key,value in word_prob.iteritems()) #assigning random probabilities
                    update_likelihood(each_topic, word_prob)
            #Test on all those remaining unread documents and find out the its topic
            for eachUnreadFIle in unreadDoc:
                words = [topicFiles[eachUnreadFIle] for topic,topicFiles in all_files.iteritems() if eachUnreadFIle in topicFiles][0]
                deducted_topic = find_topic_Unsupervised(words)
                if eachUnreadFIle not in doc_topic[deducted_topic]:
                    [doc_topic[dt].remove(eachUnreadFIle) for dt,flist in doc_topic.iteritems() if eachUnreadFIle in flist]
                    doc_topic[deducted_topic].append(eachUnreadFIle)
        print "learning got completed in ",(time.time()-sU1),"secs"    
    os.chdir(org_path)
    fileW = open(model_fileName,'w')
    pickle.dump(likelihood, fileW)
    fileW.close()
    #top 10 words with highest probability for each topic
    os.chdir(org_path)
    print "For top 10 words with highest probability, kindly check the file distinctive_words.txt"
    f = open('distinctive_words.txt','w')
    for each_topic in likelihood.keys():
        sorted_x = sorted(likelihood[each_topic].items(), key=operator.itemgetter(1),reverse =True)
        f.write("The top 10 words in topic - ")
        f.write(each_topic)
        f.write('\n')
        for i in range(0,10):
            f.write(sorted_x[i][0])
            f.write('\t')
        f.write('\n')
        f.write('\n')
    f.close()
if arguments[1]=='test':
    #print "Training complete"
    confusion=[[0]*20 for r in range(0,20)]
    fileR = open(model_fileName,'r')
    likelihood  = pickle.load(fileR)
    os.chdir(org_path+'/test')
    path = str(os.getcwd())
    topics = os.listdir(path)
    total=[each for each in topics]
    accurate = 0
    no_of_files =0
    for each_topic in topics:
        filepath = path+'/'+each_topic
        os.chdir(filepath)
        filenames = os.listdir(filepath)
        actual =[]
        prediction =[]
        #print each_topic,filenames
        for eachFile in filenames:
            no_of_files+=1
            words = list(OrderedDict.fromkeys(cleandata(convert_to_lower(readTestFile(eachFile)))))
            posterior =  dict((key,sum(value[everyWord] if everyWord in value else math.log(0.0000001) for everyWord in words)) for key,value in likelihood.iteritems())
            deducted_topic = max(posterior, key=posterior.get)
            print no_of_files,"For file ",eachFile," Deducted topic:",deducted_topic," Actual topic: ",each_topic
            if deducted_topic == each_topic:
                accurate+=1
            confusion[total.index(each_topic)][total.index(deducted_topic)]+=1
    print "Accuracy - ",(float(accurate)/no_of_files)*100
    print "Confusion Matrix -- "
    for i in range(0,20):
        print total[i]," : ",confusion[i]

