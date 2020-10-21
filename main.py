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
import csv
from Callback import Callback

#Used to calculate loss at the end of all epochs
previous_loss = 0
loss = 0

"""
Question 1: These function were used to create the graph for the first question
"""
def trainModelPerStep(path, nbrLines=1000):

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

def drawTimeCurve(lines, time):
    print("Creation de la courbe")
    plt.plot(lines, time, label="Temps en secondes")
    plt.legend(loc="upper left")
    plt.xlabel("Nombre de commentaires traitées")
    plt.ylabel("Temps passé en secondes")
    plt.title("Graphique représentant les commentaires traitées en fonction du temps")
    plt.savefig("time.png")
    plt.show()


"""
Question 2: These functions were used for question 2
"""

def computeBaseModel(filePath, step=10000, maxLines=100001):
    treated_lines = [0]
    time_array = [0]

    for i in range(maxLines):
        if i % step == 0 and i != 0:
            depart_time = time.time()
            print("Ligne " + str(i) + " sur " + str(maxLines))
            partialCorpus = MyPartialCorpus(filePath, i)
            model = gensim.models.Word2Vec(sentences=partialCorpus, workers=multiprocessing.cpu_count() * 2)
            treated_lines.append(i)
            print("Temps pour entrainer le model: " + str(time.time() - depart_time))
            time_array.append(time.time() - depart_time)

    writeBaseModel(treated_lines, time_array)
    """ No need for this part because we are using round number    
    print("Ligne " + str(maxLines) + " sur " + str(maxLines))
    # Pour finir jusqu'aux dernieres lignes
    depart_time = time.time()
    partialCorpus = MyPartialCorpus(filePath, maxLines)
    model = gensim.models.Word2Vec(sentences=partialCorpus, workers=multiprocessing.cpu_count() * 2)
    treated_lines.append(maxLines)
    print("Temps pour entrainer le model: " + str(time.time() - depart_time))
    time_array.append(time.time() - depart_time)
    """

def writeBaseModel(treated_lines, time_array):

    with open('normal_model.csv', mode='w') as data_file:
        data_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i, elem in enumerate(treated_lines):
            #Format nbrLine, temps
            data_writer.writerow([elem, int(time_array[i])])

def computeNewModels(filePath, name, step=10000, maxLines=100001, size=100, min_count=5, iter=5, sg=0, negative = 5):

    treated_lines = [0]
    time_array = []
    loss = 0

    for i in range(maxLines):
        if i % step == 0 and i != 0:
            depart_time = time.time()
            print("Ligne " + str(i) + " sur " + str(maxLines))
            partialCorpus = MyPartialCorpus(filePath, i)

            loss_callback = Callback()

            model = gensim.models.Word2Vec(sentences=partialCorpus, workers=multiprocessing.cpu_count(),
                                           size=size, min_count=min_count, iter=iter, sg=sg, negative=negative, compute_loss=True, callbacks=[loss_callback])
            loss = loss_callback.loss_previous_step/(maxLines-1)
            print("Temps pour entrainer le model: " + str(time.time() - depart_time))
            time_array.append(time.time() - depart_time)

    writeNewModel(treated_lines, time_array, name)
    return loss


def writeNewModel(treated_lines, time_array, name):

    with open(("_".join(name.split(" ")).lower())+".csv", mode='w+') as data_file:
        data_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i, elem in enumerate(treated_lines):
            #Format nbrLine, temps
            data_writer.writerow([elem, int(time_array[i])])

    return

"""
Return a list with format ([treated lines], [time])
"""
def readModel(file):
    treated_lines = []
    time_array = []
    f = open(file, "r")
    for line in f:
        line_array = line.split(',')
        treated_lines.append(line_array[0])
        time_array.append(int(line_array[1]))

    return (treated_lines, time_array)


"""
lines: array containing every lines at which we took measure of time
time_arrays = list of list containing times
labels = list of labels respectively to time_arrays
title = title of curve which will be title of png
"""
def drawComparaisonCurve(lines, time_arrays, labels, title, png_title):

    for i, elem in enumerate(time_arrays):
        plt.plot(lines, elem, label=labels[i])

    plt.legend(loc="upper left")
    plt.xlabel("Nombre de lignes traitées")
    plt.ylabel("Temps en secondes")
    plt.title(title)
    plt.savefig(("_".join(png_title.split(" ")).lower())+".png")
    plt.show()

"""
possible_values: array containing every possible value of the variable we are testing (Needs to be in the same order as time_arrays)
time_arrays = list of list containing times
title = title of curve which will be title of png
xlabel = name of the variable we are testing
"""
def drawBarPlot(possible_values, time_arrays, title, png_title, xlabel):

    for i, elem in enumerate(time_arrays):
        plt.plot(possible_values, elem)

    plt.legend(loc="upper left")
    plt.xlabel("Valeur de: " + xlabel)
    plt.ylabel("Temps en secondes")
    plt.title(title)
    plt.savefig(("_".join(png_title.split(" ")).lower())+".png")
    plt.show()


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

def generate_negative(path):
    for i in range(0, 21):
        computeNewModels(path, "negative_" + str(i), step=200000, maxLines=200001, negative=i)

    return

def generate_min_count(path):
    for i in range(0, 16):
        computeNewModels(path, "count_" + str(i), step=200000, maxLines=200001, min_count=i)
    return

def generate_iter(path):
    for i in range(0, 10):
        computeNewModels(path, "iter_" + str(i + 1), step=200000, maxLines=200001, iter=i)
    return

def generate_sg_cbow_comparaison(path):
    computeNewModels(path, "skip_gram", step=200000, maxLines=200001, sg=1, iter=4)
    computeNewModels(path, "CBOW", step=200000, maxLines=200001, sg=0, iter=4)
    return

if __name__ == '__main__':
    # Important for debug:
    # Note to advanced users: calling Word2Vec(sentences, iter=1) will run two passes over the sentences iterator (or, in general iter+1 passes; default iter=5).
    # from https://rare-technologies.com/word2vec-tutorial/#app
    # Using this information to count how many sentences have been processed

    # Loads Model From Scratch
    path = './blog/train'
    print("Processing Corpus...")
    corpus = MyCorpus(path)
    print("Generating Word2Vec Model...")
    model = gensim.models.Word2Vec(sentences=corpus, workers=multiprocessing.cpu_count(), min_count=10, iter=7, sg=0, negative=12)
    print("Calculating Similarity")
    get_similarity("./voisins.txt", model)

