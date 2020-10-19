import csv
import os
import time

from gensim.test.utils import datapath
from gensim import utils
import gensim.models
from MyCorpus import MyCorpus
import cython



"""
The CSV file is always the same. A line is always : id,  sex, age, sign, commment
Because we just want the comment, we go the the forth comma and then we take everything until the
end of the line

def readCsv (file):
    f = open(file, "r")
    sentences = []

    for line in f:
        commaCounter = 0
        sentence = ""

        for char in line:
            if commaCounter >= 4:
                sentence += char

            if(char == ","):
                commaCounter += 1

        sentences.append(sentence)

    return sentences
    


# from https://mkyong.com/python/python-how-to-list-all-files-in-a-directory/
def getCsvListFromFolder(path):
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.csv' in file:
                files.append(os.path.join(r, file))
    return files

csv_list = getCsvListFromFolder("/Users/quentinwolak/Desktop/Cours/UdeM/quatrieme_annee/IFT6285/devoir_5/blog.nosync/train")

for files in csv_list:
    comments = readCsv(files)
    for elem in comments:
        print(elem)

"""

"""
Question 3: Gets the words from the voisins.txt file and finds the 10 closest 
words using the Word2Vec Model
"""
def get_similarity(filepath, word2vec_model):
    tsv_file = open(filepath)
    read_tsv = csv.reader(tsv_file, delimiter="\t")

    with open('./word_similarity.tsv', 'w+') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for row in read_tsv:
            word = row[0]

            #Used to handle out of vocabulary errors
            try:
                #Maybe we need to sort this list???
                most_similar = word2vec_model.similar_by_word(word, topn=10, restrict_vocab=None)
            except:
                most_similar = []

            most_similar.insert(0, word)
            tsv_writer.writerow(most_similar)

    return


if __name__ == '__main__':
    # Important for debug:
    # Note to advanced users: calling Word2Vec(sentences, iter=1) will run two passes over the sentences iterator (or, in general iter+1 passes; default iter=5).
    # from https://rare-technologies.com/word2vec-tutorial/#app

    #Loads Model From Scratch
    sentences = MyCorpus('./blog/train')
    model = gensim.models.Word2Vec(sentences=sentences, workers=12)
    model.save("word2vec.model")

    #Loads saved model
    #model = gensim.models.Word2Vec.load("word2vec.model")

    get_similarity("voisins.txt", model)

    for i, word in enumerate(model.wv.vocab):
        if i == 10:
            break
        print(word)