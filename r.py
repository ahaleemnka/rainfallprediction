from Tkinter import *
import pandas as pd
from sklearn.svm import LinearSVC,SVC
from sklearn import preprocessing
from random import shuffle
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text  import CountVectorizer
from sklearn.feature_extraction.text  import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn import tree
import numpy as np
import warnings
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, f1_score,recall_score,precision_score,accuracy_score
from sklearn.model_selection import KFold

canvas_width = 1000
canvas_height =1000
warnings.filterwarnings('ignore')

true_labels=[]
slabels=[]
dlabels=[]

classifiersvm=SVC()
classifierdt=tree.DecisionTreeClassifier()

def predictt() :
	a = ent1.get()
	b=float(a)
	c = ent2.get()
	d=float(c)
	e = ent3.get()
	f=float(e)
	g = ent4.get()
	h=float(g)

	#ent4.delete(0, END)
	#ent4.insert(0,a)
	if b>94 or b<16 :
		ent5.delete(0, END)
		ent5.insert(0,"invalid input")
	if d>100 or d<15 :
		ent5.delete(0, END)
		ent5.insert(0,"invalid input")	
	if f>60 or f<0 :
		ent5.delete(0, END)
		ent5.insert(0,"invalid input")

	if h>50 or h<0 :
		ent5.delete(0, END)
		ent5.insert(0,"invalid input")
	pred_data=[[b,d,f,h]]
	c1=classifiersvm.predict(pred_data)
	c2=classifierdt.predict(pred_data)
	s=confusion_matrix(true_labels,slabels, labels=['no', 'light', 'moderate', 'heavy'])
	d=confusion_matrix(true_labels,dlabels, labels=['no', 'light', 'moderate', 'heavy'])
	print s
	print d
	flag=0
	predicitons=[]
	for i in range(0,len(c1)):
		pesvm_edt=0.0
		pedt_svm=0.0
		if c1[i]==c2[i]:
			c5=c1[i]
			predicitons.append(c5)
		else:
			if c1[i]=='no':
				if c2[i]=='light':
					n_asvmt=s[0][0]+s[0][1]+s[0][2]+s[0][3]
					n_edt=s[0][1]
					if n_asvmt !=0:
						pesvm_edt=float(n_edt)/n_asvmt
					else:
						pesvm_edt=0
				elif c2[i]=='moderate':
					n_asvmt=s[0][0]+s[0][1]+s[0][2]+s[0][3]
					n_edt=s[0][2]
					if n_asvmt !=0:
						pesvm_edt=float(n_edt)/n_asvmt
					else:
						pesvm_edt=0
				elif c2[i]=='heavy':
					n_asvmt=s[0][0]+s[0][1]+s[0][2]+s[0][3]
					n_edt=s[0][3]
					if n_asvmt !=0:
						pesvm_edt=float(n_edt)/n_asvmt
					else:
						pesvm_edt=0
				else:
					flag=1
			elif c1[i]=="light":
				if c2[i]=="no":
					n_asvmt=s[1][0]+s[1][1]+s[1][2]+s[1][3]
					n_edt=s[1][0]
					if n_asvmt !=0:
						pesvm_edt=float(n_edt)/n_asvmt
					else:
						pesvm_edt=0
				elif c2[i]=="moderate":
					n_asvmt=s[1][0]+s[1][1]+s[1][2]+s[1][3]
					n_edt=s[1][2]
					if n_asvmt !=0:
						pesvm_edt=float(n_edt)/n_asvmt
					else:
						pesvm_edt=0
				elif c2[i]=="heavy":
					n_asvmt=s[1][0]+s[1][1]+s[1][2]+s[1][3]
					n_edt=s[1][3]
					if n_asvmt !=0:
						pesvm_edt=float(n_edt)/n_asvmt
					else:
						pesvm_edt=0
				else:
					flag=1
			elif c1[i]=="moderate":
				if c2[i]=="no":
					n_asvmt=s[2][0]+s[2][1]+s[2][2]+s[2][3]
					n_edt=s[2][0]
					if n_asvmt !=0:
						pesvm_edt=float(n_edt)/n_asvmt
					else:
						pesvm_edt=0
				elif c2[i]=="light":
					n_asvmt=s[2][0]+s[2][1]+s[2][2]+s[2][3]
					n_edt=s[2][1]
					if n_asvmt !=0:
						pesvm_edt=float(n_edt)/n_asvmt
					else:
						pesvm_edt=0
				elif c2[i]=="heavy":
					n_asvmt=s[2][0]+s[2][1]+s[2][2]+s[2][3]
					n_edt=s[2][3]
					if n_asvmt !=0:
						pesvm_edt=float(n_edt)/n_asvmt
					else:
						pesvm_edt=0
				else:
					flag=1
			else:
				if c2[i]=="no":
					n_asvmt=s[3][0]+s[3][1]+s[3][2]+s[3][3]
					n_edt=s[3][0]
					if n_asvmt !=0:
						pesvm_edt=float(n_edt)/n_asvmt
					else:
						pesvm_edt=0
				elif c2[i]=="light":
					n_asvmt=s[3][0]+s[3][1]+s[3][2]+s[3][3]
					n_edt=s[3][1]
					if n_asvmt !=0:
						pesvm_edt=float(n_edt)/n_asvmt
					else:
						pesvm_edt=0
				elif c2[i]=="moderate":
					n_asvmt=s[3][0]+s[3][1]+s[3][2]+s[3][3]
					n_edt=s[3][1]
					if n_asvmt !=0:
						pesvm_edt=float(n_edt)/n_asvmt
					else:
						pesvm_edt=0
				else:
					flag=1


			if c2[i]=='no':
				if c1[i]=='light':
					n_edtt=d[0][0]+d[0][1]+d[0][2]+d[0][3]
					n_asvm=d[0][1]
					if n_edtt !=0:
						pedt_svm=float(n_asvm)/n_edtt
					else:
						pedt_svm=0
				elif c1[i]=='moderate':
					n_edtt=d[0][0]+d[0][1]+d[0][2]+d[0][3]
					n_asvm=d[0][2]
					if n_edtt !=0:
						pedt_svm=float(n_asvm)/n_edtt
					else:
						pedt_svm=0
				elif c1[i]=='heavy':
					n_edtt=d[0][0]+d[0][1]+d[0][2]+d[0][3]
					n_asvm=d[0][3]
					if n_edtt !=0:
						pedt_svm=float(n_asvm)/n_edtt
					else:
						pedt_svm=0
				else:
					flag=1
			elif c2[i]=="light":
				if c1[i]=="no":
					n_edtt=d[1][0]+d[1][1]+d[1][2]+d[1][3]
					n_asvm=d[1][0]
					if n_edtt !=0:
						pedt_svm=float(n_asvm)/n_edtt
					else:
						pedt_svm=0
				elif c1[i]=="moderate":
					n_edtt=d[1][0]+d[1][1]+d[1][2]+d[1][3]
					n_asvm=d[1][2]
					if n_edtt !=0:
						pedt_svm=float(n_asvm)/n_edtt
					else:
						pedt_svm=0
				elif c1[i]=="heavy":
					n_edtt=d[1][0]+d[1][1]+d[1][2]+d[1][3]
					n_asvm=d[1][3]
					if n_edtt !=0:
						pedt_svm=float(n_asvm)/n_edtt
					else:
						pedt_svm=0
				else:
					flag=1
			elif c2[i]=="moderate":
				if c1[i]=="no":
					n_edtt=d[2][0]+d[2][1]+d[2][2]+d[2][3]
					n_asvm=d[2][0]
					if n_edtt !=0:
						pedt_svm=float(n_asvm)/n_edtt
					else:
						pedt_svm=0
				elif c1[i]=="light":
					n_edtt=d[2][0]+d[2][1]+d[2][2]+d[2][3]
					n_asvm=d[2][1]
					if n_edtt !=0:
						pedt_svm=float(n_asvm)/n_edtt
					else:
						pedt_svm=0
				elif c1[i]=="heavy":
					n_edtt=d[2][0]+d[2][1]+d[2][2]+d[2][3]
					n_asvm=d[2][3]
					if n_edtt !=0:
						pedt_svm=float(n_asvm)/n_edtt
					else:
						pedt_svm=0
				else:
					flag=1
			else:
				if c1[i]=="no":
					n_edtt=d[3][0]+d[3][1]+d[3][2]+d[3][3]
					n_asvm=d[3][0]
					if n_edtt !=0:
						pedt_svm=float(n_asvm)/n_edtt
					else:
						pedt_svm=0
				elif c1[i]=="light":
					n_edtt=d[3][0]+d[3][1]+d[3][2]+d[3][3]
					n_asvm=d[3][1]
					if n_edtt !=0:
						pedt_svm=float(n_asvm)/n_edtt
					else:
						pedt_svm=0
				elif c1[i]=="moderate":
					n_edtt=d[3][0]+d[3][1]+d[3][2]+d[3][3]
					n_asvm=d[3][1]
					if n_edtt !=0:
						pedt_svm=float(n_asvm)/n_edtt
					else:
						pedt_svm=0
				else:
					flag=1

			if(pesvm_edt>pedt_svm):
				c5=(c1[i])
				predicitons.append(c5)
			else: 
				c5=(c2[i])
				predicitons.append(c5)
		if flag==1:
			print 'error found'
			flag=0
	print 'predcited values :',predicitons
	ent5.insert(0,predicitons[0]+'  rainfall')
	return

def mains():
	with open('hourlyxpro.csv','r') as ip:
		data=ip.readlines()
	header,rest=data[0],data[1:]
	shuffle(rest)
	with open ('hourlyxpro.csv','w') as out:
		out.write(''.join([header]+rest))
	test = pd.read_csv("hourlyxpro.csv")
	num=len(test["Temperature"])
	x_train=[]
	y_train=[]
	x_test=[]
	y_test=[]
	bins = (-1,0,1.5,3,20)
	group_names = ['no', 'light', 'moderate', 'heavy']
	categories = pd.cut(test.Precipitation, bins, labels=group_names)
	test.Precipitation = categories

	for i in xrange(0,num):
		temperature=test["Temperature"][i]
		humidity=test["RelativeHumidity"][i]
		sunlight=test["Sunlight"][i]
		wind=test["Wind"][i]
		rain=test["Precipitation"][i]
		x_train.append([temperature,humidity,sunlight,wind])
		y_train.append(rain)

	

	k_fold=KFold(n_splits=10)
	scores_svm=[]
	accu_svm=[]
	pre_svm=[]
	rec_svm=[]
	scores_dt=[]
	accu_dt=[]
	pre_dt=[]
	rec_dt=[]
	scores=[]
	accu=[]
	pre=[]
	rec=[]
	for train_indices, test_indices in k_fold.split(x_train):
		train_text=[]
		train_y=[]
		test_text=[]
		test_y=[]
		predicitons=[]
		for i in train_indices:
			temperature=test["Temperature"][i]
			humidity=test["RelativeHumidity"][i]
			sunlight=test["Sunlight"][i]
			wind=test["Wind"][i]
			rain=test["Precipitation"][i]
			train_text.append([temperature,humidity,sunlight,wind])
			train_y.append(rain)
		
		for i in test_indices:
			temperature=test["Temperature"][i]
			humidity=test["RelativeHumidity"][i]
			sunlight=test["Sunlight"][i]
			wind=test["Wind"][i]
			rain=test["Precipitation"][i]
			test_text.append([temperature,humidity,sunlight,wind])
			test_y.append(rain)
			true_labels.append(rain)
		
		classifiersvm.fit(train_text,train_y)
		predictions_svm=classifiersvm.predict(test_text)
		c1=classifiersvm.predict(test_text)
		accus=accuracy_score(test_y, predictions_svm)
		accu_svm.append(accus)
		#print '\n',accus
		pres=precision_score(test_y, predictions_svm, average='macro')
		pre_svm.append(pres)
		recs= recall_score(test_y, predictions_svm, average='macro')
		rec_svm.append(recs)
		score = f1_score(test_y, predictions_svm,average='macro')
		scores_svm.append(score)
		for j in range(0,len(predictions_svm)):
			w=predictions_svm[j]
			slabels.append(w)
		
		classifierdt.fit(train_text,train_y)
		predictions_dt=classifierdt.predict(test_text)
		c2=classifierdt.predict(test_text)
		accus=accuracy_score(test_y, predictions_dt)
		accu_dt.append(accus)
		pres=precision_score(test_y, predictions_dt, average='macro')
		pre_dt.append(pres)
		recs= recall_score(test_y, predictions_dt, average='macro')
		rec_dt.append(recs)
		score = f1_score(test_y, predictions_dt,average='macro')
		scores_dt.append(score)
		for j in range(0,len(predictions_dt)):
			w=predictions_dt[j]
			dlabels.append(w)

		s=confusion_matrix(test_y,c1, labels=['no', 'light', 'moderate', 'heavy'])
		d=confusion_matrix(test_y,c2, labels=['no', 'light', 'moderate', 'heavy'])
		flag=0
		for i in range(0,len(c1)):
			pesvm_edt=0.0
			pedt_svm=0.0
			if c1[i]==c2[i]:
				c5=c1[i]
				predicitons.append(c5)
			else:
				if c1[i]=='no':
					if c2[i]=='light':
						n_asvmt=s[0][0]+s[0][1]+s[0][2]+s[0][3]
						n_edt=s[0][1]
						if n_asvmt !=0:
							pesvm_edt=float(n_edt)/n_asvmt
						else:
							pesvm_edt=0
					elif c2[i]=='moderate':
						n_asvmt=s[0][0]+s[0][1]+s[0][2]+s[0][3]
						n_edt=s[0][2]
						if n_asvmt !=0:
							pesvm_edt=float(n_edt)/n_asvmt
						else:
							pesvm_edt=0
					elif c2[i]=='heavy':
						n_asvmt=s[0][0]+s[0][1]+s[0][2]+s[0][3]
						n_edt=s[0][3]
						if n_asvmt !=0:
							pesvm_edt=float(n_edt)/n_asvmt
						else:
							pesvm_edt=0
					else:
						flag=1
				elif c1[i]=="light":
					if c2[i]=="no":
						n_asvmt=s[1][0]+s[1][1]+s[1][2]+s[1][3]
						n_edt=s[1][0]
						if n_asvmt !=0:
							pesvm_edt=float(n_edt)/n_asvmt
						else:
							pesvm_edt=0
					elif c2[i]=="moderate":
						n_asvmt=s[1][0]+s[1][1]+s[1][2]+s[1][3]
						n_edt=s[1][2]
						if n_asvmt !=0:
							pesvm_edt=float(n_edt)/n_asvmt
						else:
							pesvm_edt=0
					elif c2[i]=="heavy":
						n_asvmt=s[1][0]+s[1][1]+s[1][2]+s[1][3]
						n_edt=s[1][3]
						if n_asvmt !=0:
							pesvm_edt=float(n_edt)/n_asvmt
						else:
							pesvm_edt=0
					else:
						flag=1
				elif c1[i]=="moderate":
					if c2[i]=="no":
						n_asvmt=s[2][0]+s[2][1]+s[2][2]+s[2][3]
						n_edt=s[2][0]
						if n_asvmt !=0:
							pesvm_edt=float(n_edt)/n_asvmt
						else:
							pesvm_edt=0
					elif c2[i]=="light":
						n_asvmt=s[2][0]+s[2][1]+s[2][2]+s[2][3]
						n_edt=s[2][1]
						if n_asvmt !=0:
							pesvm_edt=float(n_edt)/n_asvmt
						else:
							pesvm_edt=0
					elif c2[i]=="heavy":
						n_asvmt=s[2][0]+s[2][1]+s[2][2]+s[2][3]
						n_edt=s[2][3]
						if n_asvmt !=0:
							pesvm_edt=float(n_edt)/n_asvmt
						else:
							pesvm_edt=0
					else:
						flag=1
				else:
					if c2[i]=="no":
						n_asvmt=s[3][0]+s[3][1]+s[3][2]+s[3][3]
						n_edt=s[3][0]
						if n_asvmt !=0:
							pesvm_edt=float(n_edt)/n_asvmt
						else:
							pesvm_edt=0
					elif c2[i]=="light":
						n_asvmt=s[3][0]+s[3][1]+s[3][2]+s[3][3]
						n_edt=s[3][1]
						if n_asvmt !=0:
							pesvm_edt=float(n_edt)/n_asvmt
						else:
							pesvm_edt=0
					elif c2[i]=="moderate":
						n_asvmt=s[3][0]+s[3][1]+s[3][2]+s[3][3]
						n_edt=s[3][1]
						if n_asvmt !=0:
							pesvm_edt=float(n_edt)/n_asvmt
						else:
							pesvm_edt=0
					else:
						flag=1


				if c2[i]=='no':
					if c1[i]=='light':
						n_edtt=d[0][0]+d[0][1]+d[0][2]+d[0][3]
						n_asvm=d[0][1]
						if n_edtt !=0:
							pedt_svm=float(n_asvm)/n_edtt
						else:
							pedt_svm=0
					elif c1[i]=='moderate':
						n_edtt=d[0][0]+d[0][1]+d[0][2]+d[0][3]
						n_asvm=d[0][2]
						if n_edtt !=0:
							pedt_svm=float(n_asvm)/n_edtt
						else:
							pedt_svm=0
					elif c1[i]=='heavy':
						n_edtt=d[0][0]+d[0][1]+d[0][2]+d[0][3]
						n_asvm=d[0][3]
						if n_edtt !=0:
							pedt_svm=float(n_asvm)/n_edtt
						else:
							pedt_svm=0
					else:
						flag=1
				elif c2[i]=="light":
					if c1[i]=="no":
						n_edtt=d[1][0]+d[1][1]+d[1][2]+d[1][3]
						n_asvm=d[1][0]
						if n_edtt !=0:
							pedt_svm=float(n_asvm)/n_edtt
						else:
							pedt_svm=0
					elif c1[i]=="moderate":
						n_edtt=d[1][0]+d[1][1]+d[1][2]+d[1][3]
						n_asvm=d[1][2]
						if n_edtt !=0:
							pedt_svm=float(n_asvm)/n_edtt
						else:
							pedt_svm=0
					elif c1[i]=="heavy":
						n_edtt=d[1][0]+d[1][1]+d[1][2]+d[1][3]
						n_asvm=d[1][3]
						if n_edtt !=0:
							pedt_svm=float(n_asvm)/n_edtt
						else:
							pedt_svm=0
					else:
						flag=1
				elif c2[i]=="moderate":
					if c1[i]=="no":
						n_edtt=d[2][0]+d[2][1]+d[2][2]+d[2][3]
						n_asvm=d[2][0]
						if n_edtt !=0:
							pedt_svm=float(n_asvm)/n_edtt
						else:
							pedt_svm=0
					elif c1[i]=="light":
						n_edtt=d[2][0]+d[2][1]+d[2][2]+d[2][3]
						n_asvm=d[2][1]
						if n_edtt !=0:
							pedt_svm=float(n_asvm)/n_edtt
						else:
							pedt_svm=0
					elif c1[i]=="heavy":
						n_edtt=d[2][0]+d[2][1]+d[2][2]+d[2][3]
						n_asvm=d[2][3]
						if n_edtt !=0:
							pedt_svm=float(n_asvm)/n_edtt
						else:
							pedt_svm=0
					else:
						flag=1
				else:
					if c1[i]=="no":
						n_edtt=d[3][0]+d[3][1]+d[3][2]+d[3][3]
						n_asvm=d[3][0]
						if n_edtt !=0:
							pedt_svm=float(n_asvm)/n_edtt
						else:
							pedt_svm=0
					elif c1[i]=="light":
						n_edtt=d[3][0]+d[3][1]+d[3][2]+d[3][3]
						n_asvm=d[3][1]
						if n_edtt !=0:
							pedt_svm=float(n_asvm)/n_edtt
						else:
							pedt_svm=0
					elif c1[i]=="moderate":
						n_edtt=d[3][0]+d[3][1]+d[3][2]+d[3][3]
						n_asvm=d[3][1]
						if n_edtt !=0:
							pedt_svm=float(n_asvm)/n_edtt
						else:
							pedt_svm=0
					else:
						flag=1

				if(pesvm_edt>pedt_svm):
					c5=(c1[i])
					predicitons.append(c5)
				else: 
					c5=(c2[i])
					predicitons.append(c5)
			if flag==1:
				print 'adeelaaayi'
				flag=0
		accus=accuracy_score(test_y, predicitons)
		accu.append(accus)
		pres=precision_score(test_y, predicitons, average='macro')
		pre.append(pres)
		recs= recall_score(test_y, predicitons, average='macro')
		rec.append(recs)
		score = f1_score(test_y, predicitons,average='macro')
		scores.append(score)


	print '\n\nSVM classification Report:'
	print classification_report(test_y, predictions_svm)
	print 'Total datas classified:',num
	print 'Accuracy:', sum(accu_svm)/len(accu_svm)
	print 'Recall:', sum(rec_svm)/len(rec_svm)
	print 'Precision:', sum(pre_svm)/len(pre_svm)
	print 'F1_Score:', sum(scores_svm)/len(scores_svm)

	print '\n\nDT classification Report:'
	print classification_report(test_y, predictions_dt)
	print 'Total datas classified:',num
	print 'Accuracy:', sum(accu_dt)/len(accu_dt)
	print 'Recall:', sum(rec_dt)/len(rec_dt)
	print 'Precision:', sum(pre_dt)/len(pre_dt)
	print 'F1_Score:', sum(scores_dt)/len(scores_dt)

	print '\n\nCombined classification Report:'
	print classification_report(test_y, predicitons)
	print 'Total datas classified:',num
	print 'Accuracy:', sum(accu)/len(accu)
	print 'Recall:', sum(rec)/len(rec)
	print 'Precision:', sum(pre)/len(pre)
	print 'F1_Score:', sum(scores)/len(scores)
	predictt
	return

master = Tk()
master.title("Rainfall Prediction")
master.configure(background="black")
w = Label(master, text="Rainfall Prediction",width=100,height=2,bg="gray",fg="white")
w.config(font=("Courier", 25))
w.pack()


w1 = Label(master,width=40,height=20,bg="black")
w1.pack(side='right')


w3 = Label(w1,text="Temperature in F",width=30,height=2,bg="black",fg="white")
w3.pack(padx=5,pady=5)

ent1 = Entry(w1,width=15)
ent1.pack(padx=5,pady=5)

w4 = Label(w1,text="Relative Humidity (%)",width=30,height=2,bg="black",fg="white")
w4.pack(padx=5,pady=5)

ent2 = Entry(w1,width=15)
ent2.pack(padx=5,pady=5)

w5 = Label(w1,text="Sunshine durantions (Minutes)",width=30,height=2,bg="black",fg="white")
w5.pack(padx=5,pady=5)

ent3 = Entry(w1,width=15)
ent3.pack(padx=5,pady=5)

w6 = Label(w1,text="Wind speed (m/hr)",width=30,height=2,bg="black",fg="white")
w6.pack(padx=5,pady=5)

ent4 = Entry(w1,width=15)
ent4.pack(padx=5,pady=15)

button = Button(w1, text='Predict', bg = "light grey", width=10, relief=RAISED,command=predictt)
button.pack()



canvas = Canvas(master, 
           width=canvas_width, 
           height=canvas_height)
canvas.pack(side='top', fill='both', expand='yes')

img = PhotoImage(file="frog.png")
canvas.create_image(1,1,anchor=NW, image=img)




#button1 = Button(canvas, text='Load Data',width=20)
#button1.pack(side='left',padx=45,pady=10)

button2 = Button(canvas, text='Train Data',width=20,command=mains)
button2.pack(side='left',padx=80,pady=20)


w6 = Label(canvas,text="status",width=20,height=1,bg="white")
w6.pack(side='left',padx=10,pady=20)


ent5 = Entry(canvas,width=35)
ent5.pack(side='left',padx=10,pady=20)


#button4 = Button(canvas, text='Accuracy Calculation',width=20)
#button4.pack(side='left',padx=60,pady=20)

#button3 = Button(canvas, text='Combine',width=20)
#button3.pack(side='left',padx=60,pady=20)



master.geometry("1500x1500")

mainloop()
