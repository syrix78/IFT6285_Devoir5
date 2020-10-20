import csv
import multiprocessing
import os
import time

from gensim.test.utils import datapath
from gensim import utils
import gensim.models
from MyCorpus import MyCorpus
from MyPartialCorpus import MyPartialCorpus
import cython
import matplotlib.pyplot as plt

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

            # Used to handle out of vocabulary errors
            try:
                # Maybe we need to sort this list???
                most_similar = word2vec_model.similar_by_word(word, topn=10, restrict_vocab=None)
            except:
                most_similar = []

            most_similar.insert(0, word)
            tsv_writer.writerow(most_similar)

    return


def drawTimeCurve(lines, time):
    print("Creation de la courbe")
    plt.plot(lines, time, label="Temps en secondes")
    plt.legend(loc="upper left")
    plt.xlabel("Nombre de lignes traitées")
    plt.ylabel("Temps passé en secondes")
    plt.title("Graphique représentant les lignes traitées en fonction du temps")
    plt.savefig("time.png")
    plt.show()

def trainModelPerStep(path, nbrLines=30000):

    totalLines = MyCorpus(path).getTotalLinesNumber() # 643256 normalement
    print("Nombre total de lignes a traiter: " + str(totalLines))
    treated_lines = [0]
    time_array = [0]

    for i in range(totalLines):
        if i % nbrLines == 0 and i != 0:
            depart_time = time.time()
            print("Ligne " + str(i) + " sur " + str(totalLines))
            partialCorpus = MyPartialCorpus(path, i)
            model = gensim.models.Word2Vec(sentences=partialCorpus, workers=multiprocessing.cpu_count() * 2)
            treated_lines.append(i)
            print("Temps pour entrainer le model: " + str(time.time() - depart_time))
            time_array.append(time.time() - depart_time)

    print("Ligne " + str(totalLines) + " sur " + str(totalLines))
    # Pour finir jusqu'aux dernieres lignes
    depart_time = time.time()
    corpus = MyCorpus(path)
    model = gensim.models.Word2Vec(sentences=corpus, workers=multiprocessing.cpu_count() * 2)
    treated_lines.append(totalLines)
    print("Temps pour entrainer le model: " + str(time.time() - depart_time))
    time_array.append(time.time() - depart_time)

    drawTimeCurve(treated_lines, time_array)


if __name__ == '__main__':
    # Important for debug:
    # Note to advanced users: calling Word2Vec(sentences, iter=1) will run two passes over the sentences iterator (or, in general iter+1 passes; default iter=5).
    # from https://rare-technologies.com/word2vec-tutorial/#app
    # Using this information to count how many sentences have been processed

    # Loads Model From Scratch
    path = './blog/train'

    trainModelPerStep(path)

    """
    corpus = MyCorpus('/Users/quentinwolak/Desktop/Cours/UdeM/quatrieme_annee/IFT6285/devoir_5/blog.nosync/train')
    partialCorpus = MyPartialCorpus('/Users/quentinwolak/Desktop/Cours/UdeM/quatrieme_annee/IFT6285/devoir_5/blog.nosync/train', 200)

    print(corpus)
    print(partialCorpus)
    counter = 0
    for elem in partialCorpus:
        counter += 1
    print(counter)
    #model = trainModelForCurves(corpus)
    #print(model)
    model2 = gensim.models.Word2Vec(sentences=partialCorpus, workers=multiprocessing.cpu_count() * 2)
    print(model2)
    #model.save("word2vec.model")

    # Loads saved model
    # model = gensim.models.Word2Vec.load("word2vec.model")

    #get_similarity("voisins.txt", model)
    """

    """
    for i, word in enumerate(model.wv.vocab):
        if i == 10:
            break
        print(word)
    """