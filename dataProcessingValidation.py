from sklearn.feature_extraction.text import CountVectorizer
from dataExtracter import getData

def getData(dim = 12500):
    global ok
    res = [[],[]]
    with open('data.csv',encoding='utf-8') as file:
        for line in file.readlines():
            data = line.split(',')
            data[0] = int(data[0][1:len(data[0])-1]) - 1
            data[1] = data[1][1:len(data[1])-1]
            data[2] = data[2][1:len(data[2])-1]
            res[data[0]].append(data[2])
    values = []
    n = 0
    while n != dim:
        values.append((0,res[0][1]))
        values.append((1,res[1][1]))
        n += 1
    return values

def getCorrectData():
    pass