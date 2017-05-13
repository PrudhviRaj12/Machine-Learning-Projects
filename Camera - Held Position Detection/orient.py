
#Instructions to run the program - 

#Nearest neighbours - python orient.py train-data.txt test-data.txt nearest

#Neural Net - python orient.py train-data.txt test-data.txt nnet 4 
#where 4 is number of hidden layers.

#Adaboost use command - python orient.py train-data.txt test-data.txt adaboost 5
#where 5 is number of decision stumps

#Best algorithm - python orient.py train-data.txt test-data.txt best


'''

After testing a model, if you want to test it on another test set,
please delete the .txt file generated before executing the second time.
Since I have used append mode, the new results would be appended to the 
old results which will cause a dimensionality mismatch.
Nearest Neighbors:

The nearest neighbors algorithm is a non - parametric model. There is no training model,
but the entire data is stored in the disk.
For each example, a distance function is used and this function computes the distance from the
current test example to every single example in the training set. Then, it chooses a value
k (usually called k- nearest neighbors), and calculates the statistical mode of all the 
k results that are closest to the current example.

This implementation takes around 2.7 seconds for each example on average and the entire accuracy
on the given test set is 71.5 %. The value of k is taken as 30 (more on this in the PDF file)

Neural Network:

References:
The backpropagation algorithm was referred from the book Machine Learning by Tom Mitchell.

First we normalize all the values in the feature vector. 
We initialize the weights from input to hidden layer and 
from hidden layer to output layer to some random values.

Since we are using stochastic gradient descent,
each example is taken at a time. For each example,
the forward propagation is done and an output is 
predicted.

Now to learn the weights of the neural network, we backpropagate 
these errors till the input layer. These new weights serve as the
weights for the next example till the algorithm terminates. 

Sigmoid function was used as activation function
for the hidden layer as well as the output layer.
A learning rate of 0.99  was used to acheive 
convergence. For a given number of hidden layers, the algorithm
runs the entire dataset through the network 9 times ( 9 passes),
and this should not take more than a minute. 

Best Mode:

The best mode can be executed as normal neural network
with 75 hidden nodes as the parameter.
When the input is given as 75, the algorithm goes in 
the best mode. A couple of changes will take place in the best mode.
The number of training examples will be reduced by 20000
and the training set of 14976 examples will be passed 100
times through the network. This should not take more than 4 minutes.
This gave an accuracy of 74.01% on the test set (698/943).


Adaboost - 

Reference - the stage value was influced by the post on http://machinelearningmastery.com/boosting-and-adaboost-for-machine-learning

First based on number os stumps, random points on the images were selected for coparison. These selected random points served as point of comparison for each pixel.
For each ensemble (0,90,180,270) these cpmarisons between pixel's RGB values were made and based on the output the classifier which gives minimum error rate was choosen as the decision stump. Based on the error the corresppnding weights for decision stumps have been updated such that next stump focusses on those points which were wrongly classified by the eprevious stump. Each decision stump was assigned the error rate.
In testing phase, each image has been passed through n decision stumps of these4 ensembles (0,90,180,270) and based on stumps of error rate, the final decision has been made based on weighted average of each stump. The enesemble with max weight is choosen as the final orientation of the image.



'''
import pickle
import time
import operator
from collections import defaultdict
import sys
import numpy as np
import random
import math
train_file = sys.argv[1]
test_file = sys.argv[2]
method = sys.argv[3]
def readfiles(filename):
    x = open(filename).read().split()
    nll = []
    begin = 0
    end = 194
    for xe in range(len(x)):
        nll.append(x[begin: end])
        begin += 194
        end+= 194
    return nll

def confusion_matrix(expected, predicted):
    expected = np.array(expected, dtype = int)
    predicted = np.array(predicted, dtype = int)
    f = [0, 90, 180, 270]
    new_dict = defaultdict(dict)
    for fe in f:
        for fee in f:
            new_dict[fe][fee] = np.sum((predicted == fe) & (expected ==fee))
    return new_dict

if method  == 'nearest':
    
    files = readfiles(train_file)
    test_files =  readfiles(test_file)
    total_files = len(files)/194
    total_test_files = len(test_files)/194

    def distance(test_file, train_file):
        distance = 0
        for f in range(2, 194):
	    if f % 3 == 1:
                value = (int(test_file[f]) - int(train_file[f]))
	        power = pow(value, 2)
	        distance += power
        return distance

    def create_dictionary(file_name):
        current_file = file_name
        dummy_dict = dict()
        dista_0, dista_90, dista_180, dista_270 = [], [], [], []
    	for f in files[0: total_files-22000]:
	    if f[1] == '0':
	    	dista_0.append(distance(current_file, f))
	    if f[1] == '90':
	    	dista_90.append(distance(current_file,f))
	    if f[1] == '180':
	    	dista_180.append(distance(current_file,f))
	    if f[1] == '270':
	    	dista_270.append(distance(current_file, f))
    	dummy_dict['0'] = dista_0
    	dummy_dict['90'] = dista_90
    	dummy_dict['180'] = dista_180
    	dummy_dict['270'] = dista_270
	
        return dummy_dict

    def create_full_list(dummy_dict, k):
    	new_list = []
    	for j in dummy_dict.keys():
            new_list += dummy_dict[j]
    	c = []
        c = sorted(new_list)[0: k]
        return c

    def predict_outputs(c, dummy_dict):
    	de = []
    	for ce in c:
            for j in dummy_dict.keys():
	        if ce in dummy_dict[j]:
	            de.append(j)
    	return de

    def single_predicted(predicted):
    	new_dict = dict()
    	for p in predicted:
	    if p not in new_dict:
	    	new_dict[p] = 1
	    else:
	    	new_dict[p] +=1
    
        return new_dict

    outputs, gt = [], []
    size = len(test_files)/194
    #size = 20
    for f in range(0, size):
        print f
        st = time.time()
    	current_file = test_files[f]
    	gt.append(current_file[1])

        new_defdict = defaultdict(dict)
    	new_defdict = create_dictionary(current_file)
        nearest_dist = []
        nearest_dist = create_full_list(new_defdict, 30)
        predicted = []
    	predicted = predict_outputs(nearest_dist, new_defdict)
    	new_dict = dict()
    	new_dict = single_predicted(predicted)
    	outputs.append(max(new_dict.iteritems(), key=operator.itemgetter(1))[0])
    
    c= 0
    for i in range(0, size):
        if gt[i] == outputs[i]:
	    c+=1

    print 'Accuracy Nearest Neighbors: ' + str(c/float(size))
    
    print 'Writing File with Outputs - Nearest: ' 
    name = open('nearest_output.txt', 'a')
    for p in range(len(outputs)):
        name.write(test_files[p][0] + ' ' + str(outputs[p]) + '\n')

    name.close()

    x =  confusion_matrix(gt, outputs)
    y = np.array([0, 0, 90, 180, 270])
    print 'Confusion Matrix Nearest Neighbors: '
    print y
    for xe in x.keys():
        nl = []
        nl.append(xe)
        for xee in x[xe].keys():
            nl.append(x[xe][xee])
        print np.array(nl)


if method == 'nnet' or method=='best':
    if method=='best':
        layers =75
    else:
        layers=int(sys.argv[4])
    files = readfiles(train_file)
    test_files =  readfiles(test_file)
    total_files = len(files)/194
    total_test_files = len(test_files)/194

    features, targets = [], []
    for f in files[0: total_files]:
        features.append(f[2:194])
        targets.append(f[1])

    features = np.array(features, dtype = float)
    targets = np.array(targets, dtype = int)

    features_test, targets_test = [], []
    for f in test_files[0: total_test_files]:
        features_test.append(f[2:194])
        targets_test.append(f[1])

    features_test = np.array(features_test, dtype = float)
    targets_test = np.array(targets_test, dtype = int)

    for f in range(0, len(features)):
        features[f] = features[f]/float(sum(features[f]))

    def one_hot_encoding(targets):
        all_targets = []
        for t in targets:
            ohe_targets = [[0 for x in range(4)] for y in range(1)]
            if t == 0:
                ohe_targets[0][0] = 1
                all_targets.append(ohe_targets)
            if t  == 90:
                ohe_targets[0][1] =1
                all_targets.append(ohe_targets)
            if t == 180:
                ohe_targets[0][2] = 1
                all_targets.append(ohe_targets)
            if t == 270:
                ohe_targets[0][3] = 1
                all_targets.append(ohe_targets)

        return all_targets

    all_targets = one_hot_encoding(targets)
    all_targets_test = one_hot_encoding(targets_test)

    all_targets = np.array(all_targets, dtype = int)
    all_targets_test = np.array(all_targets_test, dtype = int)

    def sigmoid(x):
        return 1/ (1 + np.exp(-x))

    def neural_network(layers):
	if layers == 75:
	    print 'Best Mode Runnning. This may take 3-4 Minutes. Please Wait..'
	    passes = 100
	    excluded_files = 20000
	else:
	    passes = 9
	    excluded_files = 0
        learning_rate = 0.99
        np.random.seed(142)
        weights_i_h = np.random.randn(192, layers)
        weights_h_o = np.random.randn(layers, 4)
        for p in range(passes):
	    print 'Pass : ' +str(p)
            for f in range(0, total_files-excluded_files):
                compute_i_h = np.dot(features[f], weights_i_h)
                activation_h = sigmoid(compute_i_h)
                compute_h_o = np.dot(activation_h, weights_h_o)
                activation_o = sigmoid(compute_h_o)

                error_o = activation_o
                delta_o = error_o * ( 1 - error_o) * (error_o - all_targets[f][0])
                delta_h = activation_h * ( 1- activation_h) * (np.dot(weights_h_o, delta_o))
                c = features[f].reshape(192, 1)
                d = delta_h.reshape(layers, 1)
                weights_i_h += -learning_rate * c * d.T
                e = delta_o.reshape(4,1)
                f = activation_h.reshape(layers, 1)
                weights_h_o += -learning_rate * f * e.T

        return weights_i_h, weights_h_o
    weights_i_h, weights_h_o =  neural_network(layers)

    def predictor(feature_test):
        compute_i_h = feature_test.dot(weights_i_h)
        activation_h = sigmoid(compute_i_h)
        compute_h_o = activation_h.dot(weights_h_o)
        activation_o = sigmoid(compute_h_o)
        return np.argmax(activation_o)

    pred_targets = []
    c = 0
    for f in range(0, total_test_files):
        features_test[f] /= float(sum(features_test[f]))
        pred_targets.append(predictor(features_test[f]))
        if predictor(features_test[f]) == np.argmax(all_targets_test[f][0]):
            c+=1
    print 'Accuracy Neural Net: ' + str(c/float(total_test_files))

    output_list = []
    for p in pred_targets:
        if p == 0:
            output_list.append(0)
        if p == 1:
            output_list.append(90)
        if p == 2:
            output_list.append(180)
        if p == 3:
            output_list.append(270)
    
    print ' Writing File with Outputs - Nnet: '
    name = open('nnet_output.txt', 'a')
    for p in range(len(pred_targets)):
        name.write(test_files[p][0] + ' ' + str(output_list[p]) + '\n')

    name.close()
    
    x =  confusion_matrix(targets_test, output_list)
    y = np.array([0, 0, 90, 180, 270])
    print ' Confusion Matrix Neural Net: '
    print y
    for xe in x.keys():
        nl = []
        nl.append(xe)
        for xee in x[xe].keys():
            nl.append(x[xe][xee])
        print np.array(nl)

def readFiles_ADA(filename):
    f = open(filename, 'r')
    linetext=f.readlines()
    
    adaBoostFilesorientation = dict( ((line.rstrip('\n').split(' ')[0] + line.rstrip('\n').split(' ')[1]), int(line.rstrip('\n').split(' ')[1])  )for line in linetext)
    adaBoostFilesRed = dict( ((line.rstrip('\n').split(' ')[0] + line.rstrip('\n').split(' ')[1]), map(int,line.rstrip('\n').split(' ')[2::3])) for line in linetext)
    adaBoostFilesGreen = dict( ((line.rstrip('\n').split(' ')[0] + line.rstrip('\n').split(' ')[1]), map(int,line.rstrip('\n').split(' ')[3::3])) for line in linetext)
    adaBoostFilesBlue = dict( ((line.rstrip('\n').split(' ')[0] + line.rstrip('\n').split(' ')[1]), map(int,line.rstrip('\n').split(' ')[4::3])) for line in linetext)
    f.close()
    return (adaBoostFilesorientation,  adaBoostFilesRed,adaBoostFilesGreen,adaBoostFilesBlue)

if method == 'adaboost':
    #Reading the test and train files in the form of dictionary,
    #where dictionary is structure is - {'filename1 : [rgb_value1, rgb_value2 ,....]}
    stump_count = int(sys.argv[4])
    s = time.time()
    
    Files = readFiles_ADA(test_file)
    orientation = Files[0]
    Red = Files[1]
    Green = Files[2]
    Blue = Files[3]
    total_files = len(orientation.keys())
    
    adaBoostTrainFiles = readFiles_ADA(train_file)
    adaBoostFilesorientation = adaBoostTrainFiles[0]
    adaBoostFilesRed = adaBoostTrainFiles[1]
    adaBoostFilesGreen = adaBoostTrainFiles[2]
    adaBoostFilesBlue = adaBoostTrainFiles[3]
    total_files = len(adaBoostFilesorientation.keys())

    
    
    #print adaBoostFilesRed
    def learn_ensemble(expectedO,pixel1,pixel2,weights):
        iRG = 0.0
        iRB = 0.0
        iGB = 0.0
        tRB=0.0
        tRG=0.0
        tGB=0.0
        incorrect_RvsG = []
        incorrect_RvsB = []
        incorrect_GvsB =[]
        correct_RvsG = []
        correct_RvsB =[]
        correct_GvsB =[]
        for eachImg,orientation in adaBoostFilesorientation.iteritems():
            if adaBoostFilesRed[eachImg][pixel1] > adaBoostFilesGreen[eachImg][pixel2]:
                tRG = tRG + weights[eachImg]
                if orientation != expectedO:
                    iRG = iRG + weights[eachImg]
                    incorrect_RvsG.append(eachImg)
                else:
                    correct_RvsG.append(eachImg)
            if adaBoostFilesRed[eachImg][pixel1] > adaBoostFilesBlue[eachImg][pixel2]:
                tRB = tRB + weights[eachImg]
                if orientation != expectedO:
                    iRB = iRB + weights[eachImg]
                    incorrect_RvsB.append(eachImg)
                else:
                    correct_RvsB.append(eachImg)
            if adaBoostFilesGreen[eachImg][pixel1] > adaBoostFilesBlue[eachImg][pixel2]:
                tGB = tGB + weights[eachImg]
                if orientation != expectedO:
                    iGB = iGB + weights[eachImg]
                    incorrect_GvsB.append(eachImg)
                else:
                    correct_GvsB.append(eachImg)
        iRG = iRG/tRG
        iRB = iRB/tRB
        iGB = iGB/tGB
        m = min(iRG ,iRB,iGB)
        if iRG==m:
            return ('RG',iRG,incorrect_RvsG,correct_RvsG)
        elif iRB == m:
            return ('RB',iRB,incorrect_RvsB,correct_RvsB)
        elif iGB== m:
            return ('GB',iGB,incorrect_GvsB,correct_GvsB)

    ensemble = {0:[],90:[],180:[],270:[]}
    # 0 degree ensemble -
    weights =dict((f,1.0/total_files) for f in adaBoostFilesorientation.keys())
    for i in range(0,stump_count):
        pixel1 = random.randint(0,63)
        pixel2 = random.randint(0,63)
        (classifier, error, incorrectList,correctList)=learn_ensemble(0,pixel1,pixel2,weights)
        #stage = 0.5 * math.log(abs((1-error)/error))
        stage = abs((1-error)/error)
        ensemble[0].append(str(pixel1)+'/'+str(pixel2)+str('+'+classifier)+'*'+str(stage))
        for each in incorrectList:
            weights[each] = weights[each]*(stage)
        for each in correctList:
            weights[each]= weights[each]*(1/stage)
        #normalizing weights -
        total_wt = sum(each for each in weights.values())
        weights = dict((k,each/float(total_wt)) for k,each in weights.iteritems())
    #print ensemble[0]

    # 90 degree ensemble 
    weights =dict((f,1.0/total_files) for f in adaBoostFilesorientation.keys())
    for i in range(0,stump_count):
        pixel1 = random.randint(0,63)
        pixel2 = random.randint(0,63)
        (classifier, error, incorrectList,correctList)=learn_ensemble(90,pixel1,pixel2,weights)
        #stage = 0.5 * math.log(abs((1-error)/error))
        stage = abs((1-error)/error)
        ensemble[90].append(str(pixel1)+'/'+str(pixel2)+str('+'+classifier)+'*'+str(stage))
        for each in incorrectList:
            weights[each] = weights[each]*(stage)
        for each in correctList:
            weights[each]= weights[each]*(1/stage)
        total_wt = sum(each for each in weights.values())
        weights = dict((k,each/float(total_wt)) for k,each in weights.iteritems())
    
    #print ensemble[90]


    # 180 degree ensemble -
    weights =dict((f,1.0/total_files) for f in adaBoostFilesorientation.keys())
    for i in range(0,stump_count):
        pixel1 = random.randint(0,63)
        pixel2 = random.randint(0,63)
        (classifier, error, incorrectList,correctList)=learn_ensemble(180,pixel1,pixel2,weights)
        #stage = 0.5 * math.log(abs((1-error)/error))
        stage = abs((1-error)/error)
        ensemble[180].append(str(pixel1)+'/'+str(pixel2)+str('+'+classifier)+'*'+str(stage))
        for each in incorrectList:
            weights[each] = weights[each]*(stage)
        for each in correctList:
            weights[each]= weights[each]*(1/stage)
        total_wt = sum(each for each in weights.values())
        weights = dict((k,each/float(total_wt)) for k,each in weights.iteritems())
    #print ensemble[180]

    # 270 degree ensemble
    weights =dict((f,1.0/total_files) for f in adaBoostFilesorientation.keys())
    for i in range(0,stump_count):
        pixel1 = random.randint(0,63)
        pixel2 = random.randint(0,63)
        (classifier, error, incorrectList,correctList)=learn_ensemble(270,pixel1,pixel2,weights)
        #stage = 0.5 * math.log(abs((1-error)/error))
        stage = abs((1-error)/error)
        ensemble[270].append(str(pixel1)+'/'+str(pixel2)+str('+'+classifier)+'*'+str(stage))
        for each in incorrectList:
            weights[each] = weights[each]*(stage)
        for each in correctList:
            weights[each]= weights[each]*(1/stage)
        total_wt = sum(each for each in weights.values())
        weights = dict((k,each/float(total_wt)) for k,each in weights.iteritems())
    #print ensemble[270]

    def findOrientation(eachImg,stump_method,pix1,pix2,stump_weight):
        if stump_method == 'RG':
            if Red[eachImg][pix1] > Green[eachImg][pix2]:        
                return stump_weight
            else:
                return 0
        elif stump_method == 'RB':
            if Red[eachImg][pix1] > Blue[eachImg][pix2]:
                return stump_weight 
            else:
                return 0        
        elif stump_method == 'GB':
            if Green[eachImg][pix1] > Blue[eachImg][pix2]:
                return stump_weight
            else:
                return 0
    accuracy = 0
    totalImages = 0
    fn = open('adaboost_output.txt','w')
    confusion=[[0]*4 for r in range(0,4)]
    total = ['0','90','180','270']
    for eachImg,orie in orientation.iteritems():
        totalImages+=1
        stumpvote0 =0.0
        total0 =0.0
        stumpvote90 =0.0
        total90=0.0
        stumpvote180 =0.0
        total180=0.0
        stumpvote270 =0.0
        total270=0.0
        voteList=[]
        #test on 0 ensemble
        for each in ensemble[0]:
            stump_method = each[each.index('+')+1:each.index('*')]
            pix1 = int(each[0:each.index('/')])
            pix2 = int(each[each.index('/')+1:each.index('+')])
            stump_weight = float(each[each.index('*')+1:])
            ans = findOrientation(eachImg,stump_method,pix1,pix2,stump_weight)
            total0 = total0 + stump_weight
            if ans !=0:
                stumpvote0 = stumpvote0 + ans
        voteList.append(stumpvote0/total0)
        #test on 90 ensemble
        for each in ensemble[90]:
            stump_method = each[each.index('+')+1:each.index('*')]
            pix1 = int(each[0:each.index('/')])
            pix2 = int(each[each.index('/')+1:each.index('+')])
            stump_weight = float(each[each.index('*')+1:])
            ans = findOrientation(eachImg,stump_method,pix1,pix2,stump_weight)
            total90 = total90 + stump_weight
            if ans !=0:
                stumpvote90 = stumpvote90 + ans
        voteList.append(stumpvote90/total90)         
        #test on 180 ensemble
        for each in ensemble[180]:
            stump_method = each[each.index('+')+1:each.index('*')]
            pix1 = int(each[0:each.index('/')])
            pix2 = int(each[each.index('/')+1:each.index('+')])
            stump_weight = float(each[each.index('*')+1:])
            ans = findOrientation(eachImg,stump_method,pix1,pix2,stump_weight)
            total180 = total180 + stump_weight
            if ans !=0:
                stumpvote180 = stumpvote180 + ans
        voteList.append(stumpvote180/total180)
        #test on 270 ensemble
        for each in ensemble[270]:
            stump_method = each[each.index('+')+1:each.index('*')]
            pix1 = int(each[0:each.index('/')])
            pix2 = int(each[each.index('/')+1:each.index('+')])
            stump_weight = float(each[each.index('*')+1:])
            ans = findOrientation(eachImg,stump_method,pix1,pix2,stump_weight)
            total270 = total270 + stump_weight
            if ans !=0:
                stumpvote270 = stumpvote270 + ans 
        voteList.append(stumpvote270/total270)
        prediction = voteList.index(max(voteList))
        fn.write(eachImg[0:eachImg.index('g')+1])
        fn.write('\t')
        if prediction == 0: 
            fn.write("0")
            if orie==0 :
                accuracy+=1
            confusion[total.index(str(orie))][total.index("0")]+=1
        if prediction == 1:
            fn.write("90")
            if orie==90 :
                accuracy+=1
            confusion[total.index(str(orie))][total.index("90")]+=1
        if prediction == 2:
            fn.write("180")
            if orie== 180 :
                accuracy+=1
            confusion[total.index(str(orie))][total.index("180")]+=1
        if prediction == 3:
            fn.write("270")
            if  orie== 270 :
                accuracy+=1
            confusion[total.index(str(orie))][total.index("270")]+=1
        fn.write('\n')
    for i in range(0,4):
        print confusion[i]
    print "Accuracy : ",(accuracy/float(totalImages))*100
    fn.close()
    
