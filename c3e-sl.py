
import \
    numpy as np  # Pacote para computação científica permitindo manipulação matricial, operações algébricas e estatísticas vetorizadas (https://www.numpy.org/)
import \
    pandas as pd  # Pacote para análise e manipulação flexível de dados com estruturas similares aos Data Frames da linguagem R (https://pandas.pydata.org/)

from numpy import savetxt

import numpy.matlib

labels = pd.read_csv('inputs/labels.dat', delim_whitespace=True, names=['Real Class Labels'])
piSet = pd.read_csv('inputs/piSet.dat', delim_whitespace=True, names=['C1', 'C2', 'C3'])
SSet = pd.read_csv('inputs/SSet.dat', delim_whitespace=True, header=None)
I = 5
alpha = 0.001

def c3esl_ensemble(piSet, SSet, I, alpha):

    N = len(piSet)
    c = len(piSet.columns.values)

    piSet = np.array(piSet)
    y = [[1] * c] * N
    y = numpy.divide(y, numpy.matlib.repmat(np.sum(y, axis=1, keepdims=True), 1, c))

    for k in range(0, I):
        for j in range(0, N):
            diffi = np.arange(0, N)
            cond = diffi != j
            t1 = np.array(SSet[j][cond])

            # http://mathesaurus.sourceforge.net/matlab-numpy.html
            # https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
            p1 = (np.transpose(t1 * np.ones([c, 1])) * y[cond, :]).sum(axis=0)
            p2 = sum(t1)

            y[j, :] = (piSet[j, :] + (2 * alpha * p1)) / (1 + 2 * alpha * p2)

    print(y)
    savetxt('y.csv', y, delimiter=',', fmt='%1.4f')

c3esl_ensemble(piSet, SSet, I, alpha)

