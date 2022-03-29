#dataset -> https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews/code
# mai exact datele de test, cele de traing erau prea multe :)

from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
import re
from nltk.stem.snowball import SnowballStemmer

def processText(text):
    return re.sub(r'[^a-z]', ' ', text.lower())

def tokenize(text):
    return [stemmer.stem(word) for word in word_tokenize(text)]

stemmer = SnowballStemmer('english')
max_features = 20000
ngram_range = (1,2)
bagOfWord = CountVectorizer(
    preprocessor = processText,
    tokenizer= tokenize,
    max_features = max_features,
    ngram_range = ngram_range

)

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
    x = bagOfWord.fit_transform(x).toarray()
    return x,y


if __name__ == '__main__':
    x, y = getMatrixValues()
    print(x.shape)
