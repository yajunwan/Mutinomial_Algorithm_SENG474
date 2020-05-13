import math
import numpy

def main():

    trdataFile = open("traindata.txt","r")
   
    trdata = trdataFile.readlines()
    trdata = [line.strip('\n') for line in trdata]

    trlabelFile = open("trainlabels.txt","r")

    trlabel = trlabelFile.readlines()
    trlabel = [line.strip('\n') for line in trlabel]
    v,prior,condprob = trainMultinomialNB(trdata,trlabel)

    testFile = open("testdata.txt","r")
    tdata = testFile.readlines()
    tdata = [line.strip('\n') for line in tdata]
    tlabelFile = open("testlabels.txt","r")
    tlabel = tlabelFile.readlines()
    tlabel = [int(line) for line in tlabel]
    
    #f = open("result.txt","w")
    c = ["0","1"]

    applyMultinomialNB(v,prior,condprob,tdata,tlabel,trdata,trlabel)    



def trainMultinomialNB(data,label):

    vocabulary = []
    for line in data:
        for element in line.split():
            if element not in vocabulary:
                vocabulary.append(element)
            else:
                pass
    #print(vocabulary)
    N = len(data)

    C = ["0","1"]
    prior = {}
    condprob = {}
    for element in C:
        Nc = label.count(element)
        amount = Nc/N
        prior[element] = amount
        textc = ""
        for index in range(N):
            if label[index] == element:
                textc = textc+data[index]+" "
            else:
                pass
        #print(textc)
        Tct = {}
        Tct_total = 0
        for word in vocabulary:
            Tct[word] = textc.split().count(word)
            #print(' '+str(Tct))
            Tct_total = Tct_total+Tct[word]
        #print(Tct_total)
        for word in vocabulary:
            condprob[(word, element)] = (Tct[word] + 1) / (Tct_total + len(vocabulary))
    return vocabulary,prior,condprob

def applyMultinomialNB(v,prior,condprob,tdata,tlabel,trdata,trlabel):
    #for traindata
    buffertrainlabel = []
    count = 0
    for line in trdata:
        w = []
        #buffertrainlabel = []
        #count = 0
        for elem in line.split():
            if elem in v:
                w.append(elem)
            else:
                pass
    #print (w)
        score = {} 
        for c in ["0","1"]:
            score[c] = math.log10(prior[c])
            for t in w:
                score[c] = score[c]+math.log10(condprob[(t,c)])
        result = numpy.argmax([score["0"],score["1"]])
        buffertrainlabel.append(result)
        #print(result)
    #print(str(len(buffertrainlabel))+"--"+str(len(trlabel)))
    for i in range(len(trlabel)):
        #print(str(buffertrainlabel[i])+"++"+str(trlabel[i]))
        if int(buffertrainlabel[i]) == int(trlabel[i]):
            count+=1
    acc = count/len(trlabel)*100
    resultfile = open("results.txt","w")
    resultfile.write("The accuracy for training data is ")
    resultfile.write(str(acc)+"%\n")


    #for testdata
    buffertestlabel = []
    count = 0
    for line in tdata:
        w = []
        #buffertestlabel = []
        #count = 0
        for elem in line.split():
            if elem in v:
                w.append(elem)
            else:
                pass
    #print (w)
        score = {} 
        for c in ["0","1"]:
            score[c] = math.log10(prior[c])
            for t in w:
                score[c] = score[c]+math.log10(condprob[(t,c)])
        result = numpy.argmax([score["0"],score["1"]])
        buffertestlabel.append(result)
        #print(result)
    #print(str(len(buffertrainlabel))+"--"+str(len(trlabel)))
    for i in range(len(tlabel)):
        #print(str(buffertrainlabel[i])+"++"+str(trlabel[i]))
        if int(buffertestlabel[i]) == int(tlabel[i]):
            count+=1
    acc = count/len(tlabel)*100
    #print(acc)
    resultfile.write("The accuracy for test data is ")
    resultfile.write(str(acc)+"%\n")

    

if __name__ == "__main__":
    main()