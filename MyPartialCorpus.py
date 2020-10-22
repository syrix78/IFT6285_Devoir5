"""
Lucas Hornung --- Quentin Wolak
"""
import os

import gensim

"""
The CSV file is always the same. A line is always : id,  sex, age, sign, commment
Because we just want the comment, we go the the forth comma and then we take everything until the
end of the line
"""
def getComment(line):
    commaCounter = 0
    sentence = ""

    for char in line:
        if commaCounter >= 4:
            sentence += char

        if (char == ","):
            commaCounter += 1

    return sentence

#from https://rare-technologies.com/word2vec-tutorial/#app
class MyPartialCorpus(object):
    def __init__(self, dirname, nbrLines=100):
        self.dirname = dirname
        self.nbrLines = nbrLines


    def __iter__(self):
        counter = 0
        for fname in os.listdir(self.dirname):
            if(counter < self.nbrLines):
                if '.csv' in fname:
                    #print(fname)
                    for line in open(os.path.join(self.dirname, fname)):
                        counter += 1
                        #print(counter)
                        if(counter > self.nbrLines):
                            break
                        else:
                            #print(counter)
                            line = getComment(line)
                            #print(line)
                            yield gensim.utils.simple_preprocess(line)

