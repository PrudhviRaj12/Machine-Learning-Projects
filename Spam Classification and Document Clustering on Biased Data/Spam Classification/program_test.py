'''
Naive Bayes Method:

References:

1) HTML Tag Remover : http://stackoverflow.com/questions/9662346/python-code-to-remove-html-tags-from-a-string
2) Writing a Dictionary to File : http://stackoverflow.com/questions/899103/writing-a-list-to-a-file-with-python
3) Christopher Manning's PPT: https://web.stanford.edu/class/cs124/lec/naivebayes.pdf
4)
There a few helper files that will help the program to retrieve, read, and clean the data from files.
Please not that there is a stop_words.txt file in the working directory that I have downloaded 
from http://www.ranks.nl/stopwords.

Training Phase:
Based on the presentations by Christoper Manning (Stanford), the priors and the likelihoods are calculated
for each directory and each word respectively.

Therefore, for each word in the vocabulary, the probability of it
being spam and not spam are calculated and stored in a dictionary.

After counting these frequencies, we divide each occurence by the 
total words (values) in the dictionary. If we consider all the words,
the vocabulary size is somewhere close to 1 million ( as far as I can remember).

These mostly contain words that occur only once or in very less frequency.
Since we have converted these words into probability of their occurance, after
testing for a few thresholds, I decided to eliminate all the words whose 
probability of occurance is less than 0.000044, and this reduces the number
of words to be tested to 3773. 

The model file is created in such a way that it stores the names of the files
that contains the file names of spam and notspam dictionary. These files are
pickle files that contain the dictionary corresponding to a given type. 

The file name priors.txt stores the prior probabilities of the dataset. 

Testing Phase:
Two feature representation models were considered. The Binary model,
where represents 1 if the words occurs in document, or 0 if not. 
The Multi model which calculates the number of times the word
occurs in a document, which is assumed to give a better estimation.

The binary vector function creates a binary vector for a document
for a spam or a not spam model. For example, this is how the model works.
To test the probability whether the document is not spam, we take the
binary vector not spam, and some breaking cases are considered.

If the word is not in the training dictionary, we skip the word.
The float value in python expands minimum upto 9.88e-324. If we 
multiply a smaller number to that, that value tends to zero.
So this value is set as a limit, with the variable name MAX.
If the current value is less the MAX, we multiply the value with 
the prior and break the loop. In the actual case, if the word occurs in 
not spam vocabulary, we multiply the present probability with it.
If it does not, we take one minus the probability of word occuring in
the spam vocabulary. This is done for the spam vector as well and across
all documents.

The multi vector does something which is almost the same. In here,
assume there a word that has probability of x. If that word occurs more than
once time in a document, it results in multiplication powered by the frequency
of occurence in the document. Since higher powers lead to values that can exceed 
the float limit, we normalize the multi vector and raise the 
probabilities to the power of the normalized values.

These both methods are done for all the documents in the test set.

Results:

The binary model gives an accuracy of 84.5% on the test set,
and the multin model gives 90.2%, which is almost a 6% increase.
But the power of multinomial representation can be seen when we look
at the confusion matrix.

The binary model gives 253 false negatives on the test set,
while the multi model gives zero false negatives, which
shows that the property of multi representation discussed above holds.

But the binary model results in a higher number of True Negatives than the
multi model. This might be because of the MAX factor we set. Since the 
prior probability of the not spam documents is more, this shifts the tone
to result the probability of not spam being high.

''' 
import os
import re
import sys
import pickle
import copy
import operator

arguments = sys.argv[1:5]

current_path = os.listdir(os.getcwd())
c_p = os.getcwd()

cp_true = c_p
def set_files(dataset_dir, c_p, mode):
    set_path = str(c_p) + '/' +mode + '/'   + dataset_dir
    file_list = []
    file_list.append(os.listdir(set_path))
    return file_list, set_path

def readfiles(file_name_list, path):
    os.chdir(path)
    file_content = []
    for filename in file_name_list[0]:
        f = open(filename, 'r')
        file_content.append(f.read())
    return file_content

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
        clean_text_12 = re.sub('nbsp', '', clean_text_11)
        clean_text_13 = re.sub('\n', '', clean_text_12)
        clean_text_14 = re.sub('\t', '', clean_text_13)
        clean_files.append(clean_text_14)
    return clean_files

os.chdir(c_p)
stop_words = open('stop_words.txt', 'r').read().split()

if arguments[0] == 'train':
    if arguments[1] == 'bayes':
	priors = []
	priors_per = []
	change =  c_p+ '/' + arguments[0]
	names =  os.listdir(c_p+ '/' + arguments[0])
	for n in names:
	    priors.append(len(os.listdir(change + '/' + n)))
	#print priors
	summ = sum(priors)   
	for p in priors:
	    priors_per.append(p/float(summ))
	#print priors_per

	for i in ['notspam', 'spam']:
	    type_name, current_path = set_files(i, c_p, arguments[0])
	    files = readfiles(type_name, current_path)
            clean_files = cleandata(files)
	   
	    total_content = []
	    for f in files:
    	        total_content += f.split()

	    def count_frequencies(word_list):
    	        dictionary = dict()
    	        for w in word_list:
		#if len(w.split()) < 15:
        	    if w not in dictionary:
           	        dictionary[w] = 1
        	    else:
           	        dictionary[w] +=1
    	        return dictionary

	    file_dict = dict()
	    file_dict = count_frequencies(total_content)

            file_dict_3_40 = dict()

	    #for f in file_dict:
    	    #    if file_dict[f] > 15:
            # 	    file_dict_3_40[f] = file_dict[f]

	    
	    #file_dict = copy.deepcopy(file_dict_3_40)
	    values = 0
	    for f in file_dict:
	    	values+= file_dict[f]

	    for f in file_dict:
    	    	file_dict[f] = (file_dict[f]/(float(values)))
	    for f in file_dict:
		if file_dict[f] >=0.000044:
		    file_dict_3_40[f] = file_dict[f]

	    file_dict = copy.deepcopy(file_dict_3_40)

	    no_sw_file_dict = dict()

	    for f in file_dict:
    	        if f not in stop_words:
        	    no_sw_file_dict[f] = file_dict[f]

	    file_name = ''
	    file_name = i + '_dict.txt'
	
	
	    os.chdir(c_p)
	    name = open(file_name, 'w')
	    content = no_sw_file_dict
	    pickle.dump(no_sw_file_dict, name)
	    name.close()

	    name = open(arguments[3], 'a')
	    name.write(file_name + '\n')
	    name.close()

	    x = sorted(no_sw_file_dict.items(), key=operator.itemgetter(1), reverse = True)
	    xy =  x[250:260]
	    print 'Top 10 ' + str(i) + ' words'
	    for x in xy:
	        print x[0]

	file_name = 'priors.txt'
	name = open(file_name, 'w')
	content = priors_per
	pickle.dump(priors_per, name)
	name.close()

if arguments[0] == 'test':
    if arguments[1] == 'bayes':
	x = 0
	priors = open('priors.txt', 'rb')
	priors_per = pickle.load(priors)
	#print priors_per
	for i in ['notspam', 'spam']:
	    cd = cp_true
            priors_of_spam = round(priors_per[1], 3)
	    priors_of_not_spam = round(priors_per[0], 3)

	    type_files, test_path = set_files(i, cd, arguments[0])
	    test_files = readfiles(type_files, test_path)
            cleaned_test_files = cleandata(test_files)

	    os.chdir(cd)
	    model_name = open(arguments[3], 'rb')
	    model = model_name.read().splitlines()
	    #if model[0] == 'spam_dict.txt' or model[1] == 'spam_dict.txt':
	    if 'spam_dict.txt' in model:
	        fil = open('spam_dict.txt', 'rb')
	        no_sw_spam = pickle.load(fil)
	    #if model[0] == 'notspam_dict.txt' or model[1] == 'notspam_dict.txt':
	    if 'notspam_dict.txt' in model:
	        fil = open('notspam_dict.txt', 'rb')
	        no_sw_not_spam = pickle.load(fil)

	    v_list = no_sw_not_spam.keys() + no_sw_spam.keys()

	    def priors(spam_set, not_spam_set):
    	        all_docs = len(spam_set) + len(not_spam_set)
    	        prior_of_spam = len(spam_set)/float(all_docs)
     	        prior_of_not_spam = len(not_spam_set)/float(all_docs)

    	        return prior_of_spam, prior_of_not_spam

	    def remove_sw(file_name):
    	        now_sw_file = []
    	        for f in file_name:
        	    if f not in stop_words:
           	        now_sw_file.append(f)
    	        return now_sw_file

	    def create_bin_vects(test_file):
    	        bin_vector_ns = dict()
  	        for n in test_file:
        	    if n in no_sw_not_spam.keys():
            	        bin_vector_ns[n] = 1
        	    else:
            	        bin_vector_ns[n] = 0

    	        bin_vector_s = dict()
    	    	for n in test_file:
        	    if n in no_sw_spam.keys():
            	        bin_vector_s[n] = 1
        	    else:
            	        bin_vector_s[n] = 0
    	    	return bin_vector_ns, bin_vector_s


	    def splitter(spam_wp_sw):
    	        nf = []
    	        for s in spam_wp_sw:
         	    nf += s.split()
    	    	return nf

	    MAX = 9.88131291682e-322
	    def binary_model(bin_vector_ns, bin_vector_s):
    	        mult = 1
    	        for b in bin_vector_ns:
        	    if b not in v_list:
            	        continue
        	    if mult < MAX:
            	         mult = mult* priors_of_not_spam
            	         break
        	    if bin_vector_ns[b] == 1:
            	        if b in no_sw_not_spam:
                	     mult = mult * no_sw_not_spam[b]
            	        else:
                	     mult = mult * (1 - no_sw_spam[b])
        	    if bin_vector_ns[b] == 0:
            	        if b in no_sw_not_spam:
                	    mult = mult * ( 1 - no_sw_not_spam[b])
            	        else:
                	    mult = mult * no_sw_spam[b]
    	        mult = mult * priors_of_not_spam

    	        mult_1 = 1
    	        for b in bin_vector_s:
        	    if b not in v_list:
            	   	continue
        	    if mult_1 < MAX:
            	   	mult_1 = mult_1 * priors_of_spam
            	   	break
        	    if bin_vector_s[b] == 1:
            	        if b in no_sw_spam:
                	    mult_1 = mult_1 * no_sw_spam[b]
            	    	else: 
                	    mult_1 = mult_1 * (1 - no_sw_not_spam[b])
        	    if bin_vector_s[b] == 0:
            	        if b in no_sw_spam:
                	    mult_1 = mult_1 * ( 1 - no_sw_spam[b])
            	    	else:
                	    mult_1 = mult_1 * no_sw_not_spam[b]

    	    	mult_1 = mult_1 * priors_of_spam
    	    	return mult, mult_1

	    def mult_vect(test_file):
    	        mult_vector =  dict()
    	        for n in test_file:
        	     if n not in mult_vector:
            	         mult_vector[n] = 1
        	     else:
            	    	mult_vector[n]+=1
    	    	return mult_vector

	    def multi_model(mult_vector):
    	    	length = float(len(mult_vector))
    	        mult_spam = 1
    	    	for m in mult_vector:
        	     if m not in v_list:
            	    	continue
        	     if m in no_sw_spam:
            	    	mult_spam = mult_spam * pow(no_sw_spam[m], mult_vector[m]/length)
        	     else:
            	    	mult_spam = mult_spam * pow(1 - no_sw_not_spam[m], (mult_vector[m]/length))
    	    	mult_spam = mult_spam * priors_of_spam

    	    	mult_not_spam = 1
    	        for m in mult_vector:
        	    if m not in v_list:
            	    	continue
        	    if m in no_sw_not_spam:
            	    	mult_not_spam = mult_not_spam * pow(no_sw_not_spam[m], mult_vector[m]/length)
        	    else:
            	    	mult_not_spam = mult_not_spam * pow(1 - no_sw_spam[m], (mult_vector[m]/length))
    	    	mult_not_spam = mult_not_spam * priors_of_not_spam

    	        return mult_spam, mult_not_spam

	    def result_checker(bin_vect_ns, bin_vect_s, multi_vector):
    	    	bin_not_spam, bin_spam = binary_model(bin_vect_ns, bin_vect_s)
    	    	mult_not_spam, mult_spam = multi_model(multi_vector)
 
    	    	if bin_spam > bin_not_spam:
        	     binary =  'spam'
    	    	else:
       	 	     binary = 'notspam'
    
    	    	if mult_spam > mult_not_spam:
       	 	     multi =  'spam'
    	    	else:
        	     multi =  'notspam'
    	        return binary, multi

	    def test(file_folder):
    	        binary, multi = [], []
    	        for f in file_folder:
        	    spam_wp_sw = remove_sw([f])
        	    bin_vect_ns, bin_vect_s = create_bin_vects(splitter(spam_wp_sw))
        	    multi_vector = mult_vect(splitter(spam_wp_sw))
        	    b, m = result_checker(bin_vect_ns, bin_vect_s, multi_vector)
        	    binary.append(b)
        	    multi.append(m)
    	        return binary, multi

	    def count(data, tag):
     	        l = len(data)
     	        b_tp, b_tn, b_fp, b_fn = 0, 0, 0, 0
		m_tp, m_tn, m_fp, m_fn = 0, 0, 0, 0
		 
     	        binary, multi = test(data)
		#print tag
		#print len(data)
		if tag == 'notspam':
     	            for b in binary:
       	 	        if b == tag:
            	    	    b_tn += 1
			else:
			    b_fp += 1
			
     	    	    for m in multi:
        	        if m ==  tag:
            	    	    m_tn +=1
			else:
			    m_fp +=1
		if tag == 'spam':
		    #print '1'
		    for b in binary:
			if b == tag:
			    b_tp+=1
			else:
			    b_fn+=1
		    for m in multi:
			if b == tag:
			    m_tp+=1
			else:
			    m_fn+=1
		#print b_tp, b_tn, b_fp, b_fn, m_tp, m_tn, m_fp, m_fn			
            	return b_tp, b_tn, b_fp, b_fn, m_tp, m_tn, m_fp, m_fn
		#return b_tn, b_fp, b_fn, b_tp, m_tn, m_fp, m_fn, m_tp
	    if i == 'notspam':
	        ns = count(cleaned_test_files, i)
		x+= len(cleaned_test_files)
	    if i == 'spam':
		s = count(cleaned_test_files, i)
		x+= len(cleaned_test_files)
	x = float(x)
	total =[0]*len(ns)
	for t in range(0, len(ns)):
	    total[t] = ns[t] + s[t]
	#print total[0:4]
	#print total[4:8]
	#print total

	val_b = round((total[0] + total[1])/x, 3)
	val_m = round((total[4] + total[5])/x, 3)
	print '\n'
	print 'Accuracy of Binary Model :' +str(val_b)
	print 'Accuracy on Multi Model:' +str(val_m)
	print'\n'
	print 'Confusion Matrix For Binary Model:'
	print 'True Positives: ' +str(total[0]) + '     False Negatives: ' + str(total[3])
	print 'False Positives: ' +str(total[2]) +  '   True Negatives: ' + str(total[1])
	print '\n'
	print 'Confusion Matrix for Multi Model:'
	print 'True Positives: ' +str(total[4]) + '     False Negatives: ' +str(total[7])
	print 'False Positives: ' +str(total[6]) + '    True Negatives: ' +str(total[5])

	

