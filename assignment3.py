import sklearn.linear_model
import sklearn.neural_network
import sklearn.svm
import sklearn.ensemble 
import sklearn.model_selection
import cPickle
import math
import numpy as np

iddic = {} #author name to id
referdic = {} #paper id to reference paper list
authordic = {} #paper id to author name list
titledic = {} #paper id to paper title list
conferencedic = {} #conference name to its order
publicdic = {} #paper id to its conference order
yeardic = {} #paper id to its public year
orderdic = {} #author name to first/second/third/other order author times

featuredic = {} #author name to its feature list
paperdic = {} #author name to his/her paper id list

referingdic = {} #author name to refer he/she times
referreddic = {} #author name to be referred by he/she times

traindic = {}
testlist = list()

timesdic = {} #author name to reference times till 2011
influencedic = {} #paper id to its reference times till 2011



fw = open('../author.txt', 'r')
data = fw.readlines()
fw.close()
for line in data:
	temp = line[:-1].split('\t')
	iddic[temp[1]] = temp[0]
	timesdic[temp[1]] = 0
	featuredic[temp[1]] = list()
	orderdic[temp[1]] = [0, 0, 0, 0]

fw = open('../paper.txt', 'r')
data = fw.readlines()
fw.close()
n = len(data)
print n
confernum = 0
cnt = 0
while cnt < n:
	if data[cnt].startswith('#*'):
		index = ''
		offset = 5
		if data[cnt+4].startswith('#index'):
			index = data[cnt+4][6:]
		else:
			index = data[cnt+3][6:]
			offset -= 1
		while titledic.has_key(index):
			index += '!'
		titledic[index] = data[cnt][2:]
		authordic[index] = data[cnt+1][2:-1].split(', ')
		yeardic[index] = 2016 - int(data[cnt+2][2:])
		if offset == 5:
			if not conferencedic.has_key(data[cnt+3][2:]):
				conferencedic[data[cnt+3][2:]] = confernum
				confernum += 1
			publicdic[index] = conferencedic[data[cnt+3][2:]]
		cnt += offset
		temp = list()
		while data[cnt].startswith('#%'):
			temp.append(data[cnt][2:])
			cnt += 1
		referdic[index] = temp	
	if cnt % 100000 == 0:
		print cnt
	cnt += 1

print 'Reading file finished.'

for k in authordic:
	cnt = 0
	for item in authordic[k]:
		if paperdic.has_key(item):
			paperdic[item].append(k)
		else:
			temp = list()
			temp.append(k)
			paperdic[item] = temp
		if cnt < 3:
			orderdic[item][cnt] += 1
			cnt += 1
		else:
			orderdic[item][3] += 1
print 'Calculating paper finished.'

for k in referdic:
	for item in referdic[k]:
		if influencedic.has_key(item):
			influencedic[item] += 1
		else:
			influencedic[item] = 1
		while authordic.has_key(item):
			for name in authordic[item]:
				if timesdic.has_key(name):
					timesdic[name] += 1
					for unit in authordic[k]:
						if referingdic.has_key(unit):
							if referingdic[unit].has_key(name):
								referingdic[unit][name] += 1
							else:
								referingdic[unit][name] = 1
						else:
							referingdic[unit] = {}
							referingdic[unit][name] = 1
						if referreddic.has_key(name):
							if referreddic[name].has_key(unit):
								referreddic[name][unit] += 1
							else:
								referreddic[name][unit] = 1
						else:
							referreddic[name] = {}
							referreddic[name][unit] = 1
			item += '!'

tempdic = {}
for k in paperdic:
	tempdic[k] = len(paperdic[k])
templist = sorted(tempdic.iteritems(), key=lambda d:d[1], reverse=True)
for i in range(10):
	print templist[i]

print 'Calculating times finished.'

'''
alllist = list()
alllist.append(iddic)
alllist.append(referdic)
alllist.append(authordic)
alllist.append(titledic)
alllist.append(conferencedic)
alllist.append(publicdic)
alllist.append(yeardic)
alllist.append(paperdic)
alllist.append(referingdic)
alllist.append(referreddic)
fw = file('dumpfile', 'w')
cPickle.dump(alllist, fw)
fw.close()
'''

binnum = 102
yearbin = 73
for k in featuredic:
	featuredic[k].append(timesdic[k])
	newdic = {} #conference name to public times
	coauthordic = {} #author name to coauthor times
	temp = list()
	for i in range(yearbin):
		temp.append(0)
	#for i in range(yearbin-1):
	#	yearnum.append(76-i)
	#yearnum = [70, 60, 50, 45, 40, 35, 30, 27, 24, 21, 18, 15, 13, 11, 10, 9, 8, 7, 6]
	for item in paperdic[k]:

		if publicdic.has_key(item):
			order = publicdic[item]
			if newdic.has_key(order):
				newdic[order] += 1
			else:
				newdic[order] = 1

		for name in authordic[item]:
			if name != k:
				if coauthordic.has_key(name):
					coauthordic[name] += 1
				else:
					coauthordic[name] = 1
		if yeardic[item] > 76:
			temp[-1] += 1
		else:
			temp[yeardic[item]-5] += 1
		'''
		for i in range(yearbin-1):
			if yeardic[item] >= yearnum[i]:
				temp[i] += 1
				break
		if yeardic[item] < yearnum[-1]:
			temp[-1] += 1
		'''
	#if len(paperdic[k]) > 0:
	#	s = s * 1.0 / len(paperdic[k])
	#featuredic[k].append(s)
	s = sum(temp)
	if s > 0:
		temp = [p * 1.0 / s for p in temp]
	featuredic[k].extend(temp)
	featuredic[k].append(len(paperdic[k]))

	temp = orderdic[k]
	s = sum(temp)
	if s > 0:
		temp = [p * 1.0 / s for p in temp]
	featuredic[k].extend(temp)

	temp = list()
	for i in range(binnum):
		temp.append(0)
	#binsize = [0, 5, 10, 20, 50, 100, 200, 300, 500, 1000]
	#papersize = [0, 5, 10, 20, 50, 100, 150, 200, 300, 500]
	origin = 0
	maxnum = 100000
	binunit = maxnum / (binnum - 2)
	binsize = list()
	for i in range(binnum - 2):
		binsize.append(binunit * (i + 1))
	maxnum = 1000
	binunit = maxnum / (binnum - 2)
	papersize = list()
	for i in range(binnum - 2):
		papersize.append(binunit * (i + 1))
	for item in paperdic[k]:
		if not influencedic.has_key(item):
			temp[0] += 1
			continue
		for i in range(binnum-2):
			if influencedic[item] <= binsize[i]:
				temp[i+1] += 1
				break
		if influencedic[item] > binsize[-1]:
			temp[binnum-1] += 1
	s = sum(temp)
	if s > 0:
		temp = [p * 1.0 / s for p in temp]
	featuredic[k].extend(temp)

	if len(newdic) != 0:
		featuredic[k].append(max(newdic.items(), key=lambda x: x[1])[0])
	else:
		featuredic[k].append(-1)

	featuredic[k].append(len(coauthordic))
	temp1 = list()
	temp2 = list()
	for i in range(binnum):
		temp1.append(0)
		temp2.append(0)
	if len(coauthordic) != 0:
		for item in coauthordic:
			if timesdic[item] == 0:
				temp1[0] += coauthordic[item]
				continue
			for i in range(binnum-2):
				if timesdic[item] <= binsize[i]:
					temp1[i+1] += coauthordic[item]
					break
			if timesdic[item] > binsize[-1]:
				temp1[binnum-1] += coauthordic[item]
		for item in coauthordic:
			if len(paperdic[item]) == 0:
				temp2[0] += coauthordic[item]
				continue
			for i in range(binnum-2):
				if len(paperdic[item]) <= papersize[i]:
					temp2[i+1] += coauthordic[item]
					break
			if len(paperdic[item]) > papersize[-1]:
				temp2[binnum-1] += coauthordic[item]
	s = sum(temp1)
	if s > 0:
		temp1 = [p * 1.0 / s for p in temp1]
	featuredic[k].extend(temp1)
	s = sum(temp2)
	if s > 0:
		temp2 = [p * 1.0 / s for p in temp2]
	featuredic[k].extend(temp2)
		#featuredic[k].append(timesdic[max(coauthordic.items(), key=lambda x: x[1])[0]])
	#else:
	#	featuredic[k].append(-1)

	temp1 = list()
	temp2 = list()
	for i in range(binnum):
		temp1.append(0)
		temp2.append(0)
	if referingdic.has_key(k):
		referingtimes = 0
		for item in referingdic[k]:
			referingtimes += referingdic[k][item]
			if timesdic[item] == 0:
				temp1[0] += referingdic[k][item]
				continue
			for i in range(binnum-2):
				if timesdic[item] <= binsize[i]:
					temp1[i+1] += referingdic[k][item]
					break
			if timesdic[item] > binsize[-1]:
				temp1[binnum-1] += referingdic[k][item]

		featuredic[k].append(referingtimes)
		for item in referingdic[k]:
			if len(paperdic[item]) == 0:
				temp2[0] += referingdic[k][item]
				continue
			for i in range(binnum-2):
				if len(paperdic[item]) <= papersize[i]:
					temp2[i+1] += referingdic[k][item]
					break
			if len(paperdic[item]) > papersize[-1]:
				temp2[binnum-1] += referingdic[k][item]

		#featuredic[k].append(timesdic[max(referingdic[k].items(), key=lambda x: x[1])[0]])
	else:
		featuredic[k].append(0)
	s = sum(temp1)
	if s > 0:
		temp1 = [p * 1.0 / s for p in temp1]
	featuredic[k].extend(temp1)
	s = sum(temp2)
	if s > 0:
		temp2 = [p * 1.0 / s for p in temp2]
	featuredic[k].extend(temp2)

	if referreddic.has_key(k):
		for item in referreddic[k]:
			if timesdic[item] == 0:
				temp1[0] += referreddic[k][item]
				continue
			for i in range(binnum-2):
				if timesdic[item] <= binsize[i]:
					temp1[i+1] += referreddic[k][item]
					break
			if timesdic[item] > binsize[-1]:
				temp1[binnum-1] += referreddic[k][item]

		for item in referreddic[k]:
			if len(paperdic[item]) == 0:
				temp2[0] += referreddic[k][item]
				continue
			for i in range(binnum-2):
				if len(paperdic[item]) <= papersize[i]:
					temp2[i+1] += referreddic[k][item]
					break
			if len(paperdic[item]) > papersize[-1]:
				temp2[binnum-1] += referreddic[k][item]

		#featuredic[k].append(timesdic[max(referreddic[k].items(), key=lambda x: x[1])[0]])
	#else:
	#	featuredic[k].append(0)
	s = sum(temp1)
	if s > 0:
		temp1 = [p * 1.0 / s for p in temp1]
	featuredic[k].extend(temp1)
	s = sum(temp2)
	if s > 0:
		temp2 = [p * 1.0 / s for p in temp2]
	featuredic[k].extend(temp2)

print 'Making feature finished.'

fw = open('../citation_train.txt', 'r')
data = fw.readlines()
fw.close()
for line in data:
	temp = line.split('\t')
	traindic[temp[1]] = int(temp[2])

ftrain = list()
ltrain = list()

for k in traindic:
	ftrain.append(featuredic[k])
	ltrain.append(traindic[k])

print max(ltrain)
print min(ltrain)
neuron = len(ftrain[0])
ftrain = np.array(ftrain)
ltrain = np.array(ltrain)
#clf = sklearn.svm.SVC()
#clf = sklearn.linear_model.LinearRegression()
#clf = sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(neuron, neuron, neuron), learning_rate='invscaling')
'''
clf = sklearn.ensemble.GradientBoostingRegressor()
param_grid = {'n_estimators':range(20,81,10), 
              'learning_rate': [0.2,0.1, 0.05, 0.02, 0.01 ], 
              'max_depth': [4, 6,8], 
              'min_samples_leaf': [3, 5, 9, 14], 
              'max_features': [0.8,0.5,0.3, 0.1]} 
estimator = sklearn.model_selection.GridSearchCV(clf, param_grid)
estimator.fit(ftrain, ltrain)
'''
clf = sklearn.ensemble.GradientBoostingRegressor(n_estimators=200)
clf.fit(ftrain, ltrain)

print 'Training model finished.'

fw = open('../citation_test.txt', 'r')
data = fw.readlines()
fw.close()
for line in data:
	temp = line[:-1].split('\t')
	testlist.append(temp[1])

ftest = list()
for k in testlist:
	ftest.append(featuredic[k])

print 'Predicting finished.'

ltest = clf.predict(np.array(ftest))
print max(ltest)
print min(ltest)
fw = open('result.txt', 'w')
cnt = 0
for k in testlist:
	fw.write(iddic[k])
	fw.write('\t')
	if ltest[cnt] >= 0:
		fw.write(str(ltest[cnt]))
	else:
		fw.write('0')
	cnt += 1
	fw.write('\n')
fw.close()
