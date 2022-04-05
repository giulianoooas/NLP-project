#dataset -> https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews/code
# mai exact datele de test, cele de traing erau prea multe :)
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
import re
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import tkinter as tk

def processText(text):
    return re.sub(r'[^a-z]', ' ', text.lower())

def tokenize(text):
    return [stemmer.stem(word) for word in word_tokenize(text)]

stemmer = SnowballStemmer('english')
max_features = 20000
ngram_range = (1,2)
norm = 'l2'
bagOfWord = CountVectorizer(
    preprocessor = processText,
    tokenizer= tokenize,
    max_features = max_features,
    ngram_range = ngram_range

)
x_train,x_test,y_train,y_test = (None for i in range(4))

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

def splitDataLabels():
    """
        return data -> sentence and label
            0 -> sad
            1 -> happy
    """
    labels = []
    data = []
    for (i,j) in getData():
        labels.append(i)
        data.append(j)
    return data,labels

def getMatrixValues():
    x,y = splitDataLabels()
    bagOfWord.fit(x)
    x = normalize(bagOfWord.transform(x).toarray(),norm)
    return x,y

def setSplitTestTrain():
    global x_train,x_test,y_train,y_test
    x, y = getMatrixValues()
    if not x_train:
        x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2)
    

"""
    Prima data vom volosi un svm liniar sa vedem acuratetea
"""

def getModel(type = 'SVM'):
    model = None
    if type == 'SVM':
        model = SVC(kernel = 'linear')
    else:
        model  = KNeighborsClassifier(9)
    return trainModel(model)

def trainModel(model):
    model.fit(x_train,y_train)
    return model

def testSvc():
    setSplitTestTrain()
    model = getModel('SVC')
    predictions = model.predict(x_test)
    print(classification_report(y_test,predictions))

def testKnn():
    setSplitTestTrain()
    model = getModel('KNN')
    predictions = model.predict(x_test)
    print(classification_report(y_test,predictions))


model = None
root = tk.Tk()
root.title('NLP project')
modelSelections = None
modelPredictions = None
prediction = None
textLabel = None
buttonPredict = None
predictions = ['Sad', 'Happy']

def setVal():
    text = prediction.get()
    pred = predict(text)
    textLabel.config(text=f'"{text}" -> {predictions[pred]}')
    prediction.delete(0, tk.END)

def generateModelSelection():
    global modelSelections
    modelSelections = tk.LabelFrame(root, text="Model selection",padx=5,pady=5)
    modelSelections.pack()
    tk.Button(modelSelections, text='KNN',command=setKNN).pack()
    tk.Button(modelSelections, text='SVM',command=setSVM).pack()

def generatePredictionSpace():
    global modelPredictions, prediction, textLabel
    modelPredictions = tk.LabelFrame(root, text='Model prediction',padx=5, pady=5)
    modelPredictions.pack()
    tk.Button(modelPredictions, text="predict",command=setVal).grid(row = 0, column = 0)
    prediction = tk.Entry(modelPredictions, justify=tk.CENTER)
    prediction.grid(row = 0, column = 1)
    textLabel = tk.Label(modelPredictions,text="")
    textLabel.grid(row= 1, column=0,columnspan=2)


def destroyPredictionSpace():
    modelPredictions.destroy()

def setKNN():
    global model
    destroyPredictionSpace()
    model = getModel('KNN')
    generatePredictionSpace()



def setSVM():
    global model
    destroyPredictionSpace()
    model = getModel('SVM')
    generatePredictionSpace()

def main():
    generateModelSelection()
    generatePredictionSpace()
    root.mainloop()

setSplitTestTrain()
model = getModel()

def predict(string):
    x = normalize(bagOfWord.transform([string]).toarray(),norm)
    print(f'data -> {x}')
    # y = model.predict(x[0])
    return 1


if __name__ == '__main__':
    main()
