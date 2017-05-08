import sklearn.linear_model
import cPickle
import numpy as np

iddic = {} #author name to id
referdic = {} #paper id to reference paper list
authordic = {} #paper id to author name list
titledic = {} #paper id to paper title list
conferencedic = {} #conference name to its order
publicdic = {} #paper id to its conference order
yeardic = {} #paper id to its public year

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
	for item in authordic[k]:
		if paperdic.has_key(item):
			paperdic[item].append(k)
		else:
			temp = list()
			temp.append(k)
			paperdic[item] = temp
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

for k in featuredic:
	featuredic[k].append(timesdic[k])
	s = 0
	newdic = {} #conference name to public times
	coauthordic = {} #author name to coauthor times
	for item in paperdic[k]:
		s += yeardic[item]

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

	if len(paperdic[k]) > 0:
		s = s * 1.0 / len(paperdic[k])
	featuredic[k].append(s)
	featuredic[k].append(len(paperdic[k]))

	temp = [0, 0, 0, 0]
	for item in paperdic[k]:
		if not influencedic.has_key(item):
			temp[0] += 1
			continue
		if influencedic[item] <= 10:
			temp[0] += 1
			continue
		if influencedic[item] <= 100:
			temp[1] += 1
			continue
		if influencedic[item] <= 1000:
			temp[2] += 1
			continue
		temp[3] += 1
	s = sum(temp)
	if s > 0:
		temp = [p * 1.0 / s for p in temp]
	featuredic[k].extend(temp)

	if len(newdic) != 0:
		featuredic[k].append(max(newdic.items(), key=lambda x: x[1])[0])
	else:
		featuredic[k].append(-1)

	featuredic[k].append(len(coauthordic))
	temp1 = [0, 0, 0, 0]
	temp2 = [0, 0, 0, 0]
	if len(coauthordic) != 0:
		for item in coauthordic:
			if timesdic[item] <= 10:
				temp1[0] += coauthordic[item]
				continue
			if timesdic[item] <= 100:
				temp1[1] += coauthordic[item]
				continue
			if timesdic[item] <= 1000:
				temp1[2] += coauthordic[item]
				continue
			temp1[3] += coauthordic[item]
		for item in coauthordic:
			if len(paperdic[item]) <= 10:
				temp2[0] += coauthordic[item]
				continue
			if len(paperdic[item]) <= 20:
				temp2[1] += coauthordic[item]
				continue
			if timesdic[item] <= 50:
				temp2[2] += coauthordic[item]
				continue
			temp2[3] += coauthordic[item]
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

	temp1 = [0, 0, 0, 0]
	temp2 = [0, 0, 0, 0]
	if referingdic.has_key(k):
		referingtimes = 0
		for item in referingdic[k]:
			referingtimes += referingdic[k][item]
			if timesdic[item] <= 10:
				temp1[0] += referingdic[k][item]
				continue
			if timesdic[item] <= 100:
				temp1[1] += referingdic[k][item]
				continue
			if timesdic[item] <= 1000:
				temp1[2] += referingdic[k][item]
				continue
			temp1[3] += referingdic[k][item]
		featuredic.append(referingtimes)
		for item in referingdic[k]:
			if len(paperdic[item]) <= 10:
				temp2[0] += referingdic[k][item]
				continue
			if len(paperdic[item]) <= 20:
				temp2[1] += referingdic[k][item]
				continue
			if timesdic[item] <= 50:
				temp2[2] += referingdic[k][item]
				continue
			temp2[3] += referingdic[k][item]
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
			if timesdic[item] <= 10:
				temp1[0] += referreddic[k][item]
				continue
			if timesdic[item] <= 100:
				temp1[1] += referreddic[k][item]
				continue
			if timesdic[item] <= 1000:
				temp1[2] += referreddic[k][item]
				continue
			temp1[3] += referreddic[k][item]
		for item in referreddic[k]:
			if len(paperdic[item]) <= 10:
				temp2[0] += referreddic[k][item]
				continue
			if len(paperdic[item]) <= 20:
				temp2[1] += referreddic[k][item]
				continue
			if timesdic[item] <= 50:
				temp2[2] += referreddic[k][item]
				continue
			temp2[3] += referreddic[k][item]
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

ftrain = np.array(ftrain)
ltrain = np.array(ltrain)
clf = sklearn.linear_model.LinearRegression()
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
