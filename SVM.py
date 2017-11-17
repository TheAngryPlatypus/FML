import numpy as np
from sklearn.svm import SVC
import csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import math
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import polynomial_kernel

reader = csv.reader(open ("spambase dataset shuffled", "rb"), delimiter = ",") 
data = list(reader)

train = data[:3000]
test = data[3000:]

X_train = []
Y_train = []
for i in range(0,len(train)):
	X_train.append(list(train[i][:-1]))
	Y_train.append(train[i][-1])

X_test = []
Y_test = []
for i in range(0,len(test)):
	X_test.append(list(test[i][:-1]))
	Y_test.append(test[i][-1])

scale= StandardScaler().fit(X_train)
X_train_scaled=scale.transform(X_train)
X_test_scaled=scale.transform(X_test)	


#score=cross_val_score(clf, X_train_scaled, Y_train, cv = 10).mean()
#print "Average Score for 10-Fold CV for C=2^6 is" + str(score)
#count=t
#for i in range(0,len(X_train_scaled)-1):
#	if (abs(clf.distance_function(X_train_scaled[i]))==1):
#		count+=1
#print "Number of support vectors on the marginal hyperplane are: "+str(count)
'''drange=range(1,5)
sv=[]
for d in drange:
	clf= SVC(kernel='poly',degree=d,C=64)
	clf.fit(X_train_scaled,Y_train)
	sv.append(len(clf.support_vectors_))
plt.plot(drange,sv)
plt.title("Avg number of Support Vectors for varying degrees")	
plt.xlabel("Degree")
plt.ylabel("Number of support vectors")
plt.show()

drange=range(1,5)
sv=[]
for d in drange:
	clf= SVC(kernel='poly',degree=d,C=64)
	clf.fit(X_train_scaled,Y_train)
	pred=clf.predict(X_test_scaled)
	sv.append((1-accuracy_score(Y_test, pred))*100)
plt.plot(drange,sv)
plt.title("Avg test error for various degrees")	
plt.xlabel("Degree")
plt.ylabel("Test error in %")
plt.show()
'''
'''
def New_Kernel(X,Y):
	temp=np.dot(X,Y.T)
	sum=np.empty(temp.shape)
	for i in range(1,5):
		for j in range(i,5):
			sum=np.add((np.dot(X,Y.T)**(i+j)),sum)
	return 0.1*sum
clf=SVC(kernel= New_Kernel, degree=1, C=2)
clf.fit(X_train_scaled,Y_train)
predict=clf.predict(X_test_scaled)
print accuracy_score(Y_test,predict)
'''
'''
def my_custom_kernel(X,Y):
	u = 4
	norm = 2.0/(u*(u+1))
	temp= my_poly_kernel(X,Y,1) 
	x = np.empty(temp.shape)	
	for j in range(u):
		for i in range(j):
			x= np.add(my_poly_kernel(X,Y,i+j+2),x)
	return norm * x 

def my_poly_kernel(X,Y,d):
	return np.dot(X, Y.T)**d
'''
'''clf=SVC(kernel= my_custom_kernel, degree=1, C=2)
clf.fit(X_train_scaled,Y_train)
predict=clf.predict(X_test_scaled)
print accuracy_score(Y_test,predict)
'''
'''clf= SVC(kernel=poly,d=1,C=64)
clf.fit(X_train_scaled,Y_train)
supp_vecs=clf.support_vectors_
dist=clf.decision_function(supp_vecs)
print dist
count=0
for i in dist:
	if abs(round(i, 3))==1.000:
		count+=1
print count
"if round not used=0, if used at 3rd place after floating point= 68."
'''

drange=range(1,5)
c=64
scores=[]
predictions=[]
for d in drange:
	clf= SVC(kernel='poly',degree=d,C=64)
	score=cross_val_score(clf, X_train_scaled, Y_train, cv = 10).mean()
	scores.append((1-score)*100)
plt.plot(drange,scores)
plt.title("Avg scores of 10-fold CV for various degrees with C=64")	
plt.xlabel("Degree")
plt.ylabel("CV error in %")
plt.show()	

'''clf = SVC(kernel='precomputed',degree=d,C=2**k)
		gram = my_custom_kernel(X_train_scaled,X_train_scaled)
'''
'''drange=range(1,4)
krange=range(5,11)
for d in drange:
	mean_scores=[]
	lstd=[]
	ustd=[]
	for k in krange:
		clf = SVC(kernel='precomputed',degree=d,C=2**k)
		gram = my_custom_kernel(X_train_scaled,X_train_scaled)		
		scores=cross_val_score(clf, gram, Y_train, cv = 10)
		score=scores.mean()
		mean_scores.append(score)
		lst=score-scores.std()
		ust=score+scores.std()
		lstd.append(lst)
		ustd.append(ust)
	plt.plot(krange,mean_scores,linewidth=2, label= "Mean")
	plt.plot(krange,lstd,'--',linewidth=1, label= "Lower_Std_Dev")
	plt.plot(krange,ustd,'--',linewidth=1, label= "Higher_Std_Dev")
	plt.title("Degree "+str(d))	
	plt.xlabel("log_2_C")
	plt.ylabel("Mean score and Std Dev")
	plt.show()
'''
