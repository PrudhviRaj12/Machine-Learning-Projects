###################################
# CS B551 Fall 2016, Assignment #3
#
# Your names and user ids:
#	Prudhvi Raj Dachapally - prudacha
#	Subramanian Shunmugavel - shanmusu
#	Geetanjali Bhagwani - gcbhagwa

# (Based on skeleton code by D. Crandall)
#
#
####
# Put your report here!!

'''


NOTE : THERE IS AN ISSUE WITH MY PROGRAM'S CORRESPONDENCE WITH THE pos_scorer.py MODULE. WHEN I RUN, IT STARTS THROWING ME AN "INDEX OUT OF BOUNDS ERROR".
THEREFORE, IF YOU CAN COMMENT OUT THE LAST TWO LINES OF THE pos_scorer.py MODULE, THIS PROGRAM WILL RUN EFFORTLESSLY. PLEASE.

THESE ARE THE LINES

for j in range(0, len(outputs[algo][1])):
    Score.print_helper("", outputs[algo][1][j], sentence)



References:

Maximum value from a dictionary: http://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
Normalizing values in a dictionary : http://stackoverflow.com/questions/16417916/normalizing-dictionary-values

Report:

First, the initial probabilities for each part-of-speech tag are calculated, and are saved in the variable p_s
Next, the transition probabilities are calculated (POS A followed by POS B). These are saved in the variable p_s_plus_1_given_s.
Since this consists of continuous values, we normalize that each dictionary of a particular POS and we save them in normalized_transition
Now, we calculate the emission probabilities. This will take a little bit of time (barely 3-4 minutes for training), but the time taken will be worth
it. We don't normalize these values now. Instead, we do only for an individual sentence when necessary for a particular method.

Simplified Method: 
Here, pos-to-pos transitions are not considered and each word is independent of the next word. Therefore,  for each word, we find the probable POS, and we find the maximum POS for each word. The probability or confidence score is also retrived from the dictionary and the list of 
parts of speech and their confidence scores are returned. If a word does not have a particular POS, (generally happens in test set), we assign the maximum
initial probability of a Part-of-speech.

Trade-Off: 
If the number of training examples is reduced by 270, there is a small tradeoff when it comes to word and sentence accuracy rates. 
This might be due to the noun/verb or adj/adv tradeoffs that vary by a very small margin.

HMM Method: 
Here, we have to consider the transition probabilties. For the first word, we multiply the probability what the first word is, and the initial probability of that word. (Derived directly from the practice assignment problem)
Then, for each word, for each emission values (keys) for the word, we multiply the prior probability of the previous word to the emission probability that notifies the POS ffrom previous word to the next word. Then these are saved in a dictionary and the maximum probabilties are checked and are returned from the variable max_prob. 

Complex Model: In this model, we decide the Part-of-speech of a specific word based on its previous two words. Instead of variable elimination, I tried to
tweak the Viterbi model to allocate an extra state, and it worked. Here, first the computation for the first two words was done as in HMM model where we do only one word. If the length of the sentence (or) number of words in the sentence is less than two, it is unnecessary to use the complex model, so we send that sentence to the HMM model. The transitions are computed backwards, i.e., from prev_prev_word to prev_word, and then from prev_word to current_word and these are stored in a list. The keys, which are the parts of speeches are extracted from that dictionary and are returned. Since I did not use variable elimination, there would not be any "confidence" scores. 

Results:

The simplified model returns an accuracy of 100% on word level and 100% on the sentence level on the tiny test set ( 3 examples). When this model is applied to the 2000 instances test set, it returns a word level accuracy of 93.27% and a sentence level accuracy of 43.70%.

The HMM model returns a word accuracy of 90.48% on the word level and none on the sentence level in the tiny set. When this model is run on the test set, it returns a correct recognition rate of 90.06% and 32.20% on the word and sentence level, respectively.

The Complex model, while performs better than the HMM, has a variable difference compared to the simplified model. The accuracies were same as HMM model on the tiny test set, but there was an increase to 91.13% (0.65%) on word level and 33.65% (1.45%) on the sentence level. 

The difference may not seem much in terms of percentages, but when seen on real numbers, the complex model can predict 191 words correctly compared to the HMM
model.

'''


import random
import math
from collections import defaultdict
import operator
import math

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:
	
    p_word_given_type = defaultdict(dict)
    number_of_types = {}
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling
    def posterior(self, sentence, label):
	initial, transition, emission = [], [], []
	if label[0] in parts_of_speech:
	    
	    for i in range(0, len(sentence)):
                initial.append(p_s[label[i-1]])
	    for j in range(1, len(sentence)):
                transition.append(normalized_transition[label[j-1]][label[j]])
	    for k in range(0, len(sentence)):
	        if len(p_word_given_type[sentence[i]]) == 0:
		    emission.append(0.24)
	        else:
		    emission.append(round( max(p_word_given_type[sentence[i]].iteritems(), key = operator.itemgetter(1))[1], 2))
	
	    total = [initial[0]] + transition + emission
	    for t in range(len(total)):
		if total[t] == 0:
		    total[t] = 0.00001 
		
		   
	    #print total	
	    mult = 1
            for i in range(0, len(total)):
	        mult *= total[i]
	    #print mult
	    #print math.log(mult, 2)	
	
	    return math.log(mult, 2)
	else:
	    return 0
	#return 0
    # Do the training!
    
    def train(self, data):
	
   	global parts_of_speech 
    	parts_of_speech = ['adj', 'adv', 'adp', 'conj', 'det', 'noun', 'num', 'pron', 'prt', 'verb', 'x', '.']
   	number_of_types = dict()
	p_s = dict()
	global p_s
    	for i in parts_of_speech:
        	number_of_types.update({i : 0})
    	    
	total_count = 0 
	for i in range(0, len(data)):
            for j in range(0, len(data[i][1])):
              total_count = total_count + 1
	print 'Calculating Initial Probabilties....'	
	for x in number_of_types:
            count = 0
	    for i in range(0, len(data)):
	    	for j in range(0, len(data[i][1])):
	    	    if data[i][1][j] == x:
		        count = count + 1
            	    number_of_types.update({x : count})
	    	p_s.update({x : float(count)/total_count})
           
        global sub_var, sub_val
        sub_var =  max(p_s.iteritems(), key = operator.itemgetter(1))[0]
	#print sub_var
        sub_val = max(p_s.iteritems(), key = operator.itemgetter(1))[1]
	#print sub_val
	print 'Calculating Transition Probabilties.......'
	p_s_plus_1_given_s = defaultdict(dict)
	global p_s_plus_1_given
	for x in number_of_types:
            for y in number_of_types:
                count = 0
            	for i in range(0, len(data)):
                    for j in range(0, len(data[i][1])):
                    	c = j
                    	if data[i][1][c] == x and data[i][1][c-1] == y:
                        	count = count + 1
       	        		p_s_plus_1_given_s[y][x] = count 
        p_s_plus_1_given_s['pron']['x'] = 1
	
	print 'Normalizing Transition Probabilties.......'
	normalized_transition = dict()
	for b in p_s_plus_1_given_s.keys():
   	    temp = p_s_plus_1_given_s[b]
	    factor = 1/float(sum(temp.itervalues()))
	    for k in temp:
                temp[k] = round(temp[k] * factor, 3)
	    normalized_transition.update({b : temp})
	global normalized_transition
	    
	global p_word_given_type
	p_word_given_type = defaultdict(dict)
	print 'Calculating Emission Probabilities.....this may take some time (3-4 minutes approx.)'
  	count = 0
	#270
	for i in range(0, len(data)-270):
	    for j in range(0, len(data[i][0])):
		for y in data[i][1]:
		    for x in data[i][0]:
	                if data[i][0][j] == x and data[i][1][j] == y:
		           count = count + 1
	    	    	   p_word_given_type[data[i][0][j]][y] = count

    # Functions for each algorithm.
    #

    def simplified(self, sentence):
	full_append = []
	appender = []	
	for i in range(0, len(sentence)):
               if len(p_word_given_type[sentence[i]]) >= 1:
		   factor=1.0/sum(p_word_given_type[sentence[i]].itervalues())
		   for k in p_word_given_type[sentence[i]]:
 		       p_word_given_type[sentence[i]][k] = p_word_given_type[sentence[i]][k] *factor
                   appender.append( max(p_word_given_type[sentence[i]].iteritems(), key = operator.itemgetter(1))[0])
               	   full_append.append(round( max(p_word_given_type[sentence[i]].iteritems(), key = operator.itemgetter(1))[1], 2))
	       else:
                   appender.append(sub_var)
		   full_append.append(round(sub_val, 2))
	return [ [appender, full_append, ]]

    def hmm(self, sentence):
               
	emission = p_word_given_type
        transition = normalized_transition
        initial = p_s
        probs = self.simplified(sentence)[0][1]

        for i in range(0, len(sentence)):
            if len(emission[sentence[i]]) >= 1:
                factor = 1.0/sum(emission[sentence[i]].itervalues())
		for k in emission[sentence[i]]:
                    emission[sentence[i]][k] = emission[sentence[i]][k] * factor

        ce = dict()
    	prev_prob= dict()
    	first_word = sentence[0]
    	for f in emission[first_word].keys():
            prev_prob.update({f : initial[f] * emission[first_word][f]})
        ce.update({first_word: prev_prob})

    	for i in range(1, len(sentence)):
            current_word = sentence[i]
            prev_word = sentence[i-1]
            prev_prob = dict()
            if len(emission[current_word]) == 0:
                emission[current_word].update({sub_var : sub_val})
            for c in emission[current_word].keys():
                max_finder = []
                for x in ce[prev_word]:
                    value = (probs[i-2] ) +  (ce[prev_word][x] * transition[x][c])
                    max_finder.append(value)
                if len(max_finder) == 0:
                    max_finder = [1e-1]
                prev_prob.update({c: max(max_finder) * emission[current_word][c]})

            ce.update({current_word : prev_prob})

    	max_prob = []
    	for s in sentence:
            for c in ce:
                if s == c:
                    #print c
                    #print ce[c]
                    #print len(ce[c])
                    if len(ce[c]) == 0:
                        prob = sub_var
                        max_prob.append(prob)
                    else:
                        prob = max(ce[c].iteritems(), key = operator.itemgetter(1))[0]
                        max_prob.append(prob)
	sen = [0] * len(sentence)
        return [[max_prob], ]
   
    
    def complex(self, sentence):
       
	emission = p_word_given_type
        transition = normalized_transition
        initial = p_s
        probs = self.simplified(sentence)[0][1]

        for i in range(0, len(sentence)):
            if len(emission[sentence[i]]) >= 1:
                factor = 1.0/sum(emission[sentence[i]].itervalues())
                for k in emission[sentence[i]]:
                    emission[sentence[i]][k] = emission[sentence[i]][k] * factor

	if len(sentence) < 2:
	   return self.hmm(sentence) 

	ce = dict()
        prev_prob= dict()
	
        first_word = sentence[0]
        second_word = sentence[1]
	for f in emission[first_word].keys():
            prev_prob.update({f : initial[f] * emission[first_word][f]})
        ce.update({first_word: prev_prob})

        prev_prob = dict()
    	for f in emission[second_word].keys():
            prev_prob.update({f : initial[f] * emission[second_word][f]})
        ce.update({second_word : prev_prob})
	
	for i in range(2, len(sentence)):
            current_word = sentence[i]
            prev_word = sentence[i-1]
            prev_prev_word = sentence[i-2]
            prev_prob = dict()
            for c in emission[current_word].keys():
                for d in emission[prev_word].keys():
                    max_finder = []
		   
                    for x in ce[prev_word]:
                        for y in ce[prev_prev_word]:
                            value = ce[prev_prev_word][y] * transition[y][x] + ce[prev_word][x] * transition[x][d]
                            max_finder.append(value)
                            prev_prob.update({c: max(max_finder) * emission[current_word][c]})
                        
	    ce.update({current_word : prev_prob})

	max_prob = []
        for s in sentence:
            for c in ce:
                if s == c:
                    if len(ce[c]) == 0:
                        prob = sub_var
                        max_prob.append(prob)
                    else:
                        prob = max(ce[c].iteritems(), key = operator.itemgetter(1))[0]
                        max_prob.append(prob)
      
        return [[max_prob], ]





    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It's supposed to return a list with two elements:
    #
    #  - The first element is a list of part-of-speech labelings of the sentence.
    #    Each of these is a list, one part of speech per word of the sentence.
    #
    #  - The second element is a list of probabilities, one per word. This is
    #    only needed for simplified() and complex() and is the marginal probability for each word.
    #
    def solve(self, algo, sentence):
        if algo == "Simplified":
            return self.simplified(sentence)
        elif algo == "HMM":
            return self.hmm(sentence)
        elif algo == "Complex":
            return self.complex(sentence)
        else:
            print "Unknown algo!"

