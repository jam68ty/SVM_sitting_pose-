from sklearn import svm
import numpy as np
from firebase import firebase
import json


with open('Posenet_Points (10).json', 'r') as f:
    jf = json.loads(f.read())
    
a=np.empty(shape=[0,5])#建一個空的array
#用檢索的方式把資料放進a裡面
for i in range(0,len(jf)):
    for j in range(0,len(jf[0]['keypoints'])):
        x=jf[i]['keypoints'][j]['position']['x']
        y=jf[i]['keypoints'][j]['position']['y']
        a=np.append(a,[x])
        a=np.append(a,[y])
        
#print(a)
b= a.flatten() #把資料讀為一維陣列
#print(b)
X = b.reshape(len(jf), 34)#每34個數字成一組
#print(X)

#---------讀txt檔---------#
#np.set_printoptions(suppress=True) #設置print輸出，預設為科學記號輸出
#a = np.genfromtxt('E:\\test.txt') #一開始讀入txt
#print("a:",a)

#b= a.flatten() #把資料讀為一維陣列
#print("b",b)

#X = b.reshape(a.shape[0], 34)
#print(a.shape[0])
#print(X)

#Test = np.genfromtxt('E:\\test2.txt')  #這裡放單筆資料
#Test=Test.reshape(1,-1) #if your data has a single feature or array.reshape(1, -1) if it contains a single sample.

#print(Test)
#----------------------------#

# A B C
A = ('Y','Y')
B = ('N','N')
Y = A+B
#Y = ('Y', 'N','Y','N')



with open('test.json', 'r') as f:
    test = json.loads(f.read())
    
b=np.empty(shape=[0,5])#建一個空的array
#用檢索的方式把資料放進b裡面
for j in range(0,len(test[0]['keypoints'])):
    x=test[0]['keypoints'][j]['position']['x']
    y=test[0]['keypoints'][j]['position']['y']
    b=np.append(b,[x])
    b=np.append(b,[y])
#print(b)
c= b.flatten() #把資料讀為一維陣列
#print(b)
test_data = c.reshape(len(test), 34)#每34個數字成一組
#print(test_data)


# ovo: one-against-one, ovr:one-vs-the-rest

cw = {'Y':2,'N':1}

sv = svm.SVC(decision_function_shape='ovr',
             C=1,
             class_weight=cw,
            kernel='rbf', gamma=0.5)
 #           kernel='linear')

sv.fit(X, Y) 
print('\n Predict for the test data =  ', sv.predict(test_data))

#把預測結果回傳至資料庫
database=sv.predict(test_data)
firebase = firebase.FirebaseApplication('https://wellsitting-ef0e5.firebaseio.com/', None)
new=database.tolist()#把nparray轉為列表(為JSON serializable)
result=firebase.put('/testForStatus/1','status',new)
#put:改寫或寫入 post:新增資料
