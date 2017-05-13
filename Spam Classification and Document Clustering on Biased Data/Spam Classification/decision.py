Training phase:
For binary part: 
Each of the top 2000 words are taken as features and for every email if the word occurs in the email then it is stored as 1
else it is stored as 0
Entropy is calculated by the formula given in class.
For each word as feature the word with minimum entropy is taken and used to split.
The word or feature is deleted then using recursion entropy of other words are calculated and the words are split
For the continous part:
Each of The top 2000 words are taken as features and for every email if the word occurs in the email then the frequency of the word is stored it 
else it The median of the word is taken for split removing the original node recurssively entropy of the other words are calculated.          

decision = {"email1":{"word1":0, "word2":0, "word3":0, "word4":1, "spam":0},
            "email2":{"word1":1, "word2":0, "word3":1, "word4":1, "spam":1},
            "email3":{"word1":1, "word2":1, "word3":1, "word4":1, "spam":1},
            "email4":{"word1":1, "word2":1, "word3":0, "word4":1, "spam":0}}
print decision
nt = len(decision)
"""
words = {}
for key1 in decision:
    #print key1, decision[key1]
    for key2 in decision[key1] :
        print key2, decision[key1][key2]
"""
for key1 in decision:
    for key2 in decision[key1]:
        spam_count = 0
        spam_count2 = 0
        if key2 is not "spam":
            search_val = decision[key1][key2]
            #print search_val
            for temp in decision[key1]:
                if temp is not "spam":
                    if decision[key1][temp] == int(search_val) and decision[key1]["spam"] == 0:
                        spam_count+=1
                    if decision[key1][temp] == int(search_val) and decision[key1]["spam"] == 1:
                        spam_count2+=1
            #print decision[key1]["spam"], decision[key1][key2]
            print decision[key1][key2], decision[key1]["spam"], spam_count, spam_count2
            entropy ={}
            entropy{key1} = {}
            entropy{key1} = decision[key1][temp] 
for key1,value1 in entropy.iteritems():
    min = min(value1, key = value1.get)


