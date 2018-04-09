import numpy as np
import pandas as pd
import random

errCollect=[]
numpyData = np.array
numpySurvived = np.array
numpyTestData = np.array

def pandasIO(filename):
    # read data
    data = pd.read_csv(filename)
    print(data.shape)

    # 淘汰用不到的值
    data = data.drop(["Name", "Ticket", "Cabin", "Embarked"], axis=1)
    if filename == "train.csv":
        # 捨棄null的資料
        data = data.dropna()
    else:
        # 填滿空白的資料
        data = data.fillna(30)

    # 原始資料加上constant
    data['Const'] = pd.Series(np.repeat(1, len(data)), index=data.index)
    # 性別轉成int
    sexConvert = {"male": 1, "female": 2}
    surviveConvert = {1: 1, 0: -1}
    data['Sex'] = data['Sex'].apply(sexConvert.get).astype(int)
    # 特徵值與乘客資料分開
    # , 'Age',
    # ,'Age' ,'SibSp','Parch',,'Fare'
    features = ['Const', 'Pclass', 'Sex','Fare']
    # 轉成 numpy
    global numpyData , numpySurvived,numpyTestData
    if filename == "train.csv":
        numpyData = data[features].as_matrix()
        numpySurvived = data['Survived'].apply(surviveConvert.get).astype(int).as_matrix()
    else:
        numpyTestData = data[features].as_matrix()
    return data


# 錯誤率計算結束
def errorCount(current):
    global numpyData, numpySurvived
    print("################ 錯誤率計算開始 ##############")
    err = 0
    count = 0
    errCollect.clear()
    for dataset in numpyData:
        t = np.dot(current, dataset)
        #print(numpySurvived[count], np.sign(t))
        if numpySurvived[count] != np.sign(t):
            err += 1
            errCollect.append(count)
        count += 1
    print("err: ",err,"\n","errcollect: ",len(errCollect))
    print("################ 錯誤率計算結束 ##############")
    return err
# 錯誤率計算結束
################# 修正權重 ######################
def reviseWeight(Weight,errCollect):
    global numpyData, numpySurvived
    print("################ 修正權重 ##############")
    num = random.randint(0, len(errCollect)-1)

    count = errCollect[num]
#    print(numpySurvived[count])
#    print(numpyData[count])
    Weight = Weight + numpySurvived[count] * numpyData[count]
    print(Weight)
    print("################ 修正權重 ##############")
    return Weight
################# 修正權重 ######################
def pocketPLA(limit):
    global numpyData, numpySurvived
    Weight = random.sample(range(80, 120), 4)  # 紀錄權重，第 0 格為 threshold
    print(Weight)
    print(len(numpyData))
    least_false = errorCount(Weight)
    for i in range(limit):
        newWeight = reviseWeight(Weight,errCollect)
        current_false = errorCount(newWeight)
        print(current_false,"  ",least_false)
        if current_false <= least_false:
            least_false = current_false
            currentBestWeight = newWeight
        Weight = newWeight
    print(least_false , Weight)
    return currentBestWeight

def Counting(data,Weight):
    global numpyTestData
    Y =[]
    print(Weight)
    for dataset in numpyTestData:
        y = np.dot(Weight, dataset)
        isSurvived = np.sign(y)
#        print("isSurvived: ",isSurvived)
        if isSurvived == -1.0:
            isSurvived = 0;
        else: isSurvived=1;
        Y.append(int(isSurvived))
#    print(Y)
    data['Survived'] = pd.Series(Y, index=data.index)
    header = ["PassengerId","Survived"]
    data.to_csv('output.csv', columns=header,index=False)

abd_data = pandasIO('train.csv')
#print(numpyData)
#print(numpySurvived)
finalWeight = pocketPLA(15000)
data = pandasIO('test.csv')
Counting (data,finalWeight)