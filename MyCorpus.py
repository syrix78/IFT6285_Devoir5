import os


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
class MyCorpus(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                line = getComment(line)
                #print(line)
                yield line.split()

