import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import math
import scipy.stats as stats
import sys
import random

MIN_RANGE = -200
MAX_RANGE = 200
NUMBER_OF_ELEMENTS = 800
ax = np.linspace(MIN_RANGE, MAX_RANGE, NUMBER_OF_ELEMENTS)
pw1 = 0.5
pw2 = 0.5
mu1 = -5
mu2 = 10
mu3 = 5
sigma = 4
prior1 = stats.norm.pdf(ax, mu1, sigma)
prior2 = stats.norm.pdf(ax, mu2, math.sqrt(2) * sigma)
prior3 = stats.norm.pdf(ax, mu3, math.sqrt(3) * sigma)

class Data:
    X = 0
    Y = 0

def gx(mu, sigma, pw):
    i = 0;
    vx = [0] * len(ax)
    for x in ax:
        vx[i] = ((-0.5) * math.log(2 * np.pi)) - math.log(sigma) - (math.pow((x - mu),2) / (2 * math.pow(sigma,2))) + math.log(pw)
        i = i+1
    return vx

def q1a():
    plt.plot(ax, gx(mu1, sigma, pw1))
    plt.plot(ax, gx(mu2, math.sqrt(2) * sigma, pw2))
    plt.show()
    return

def q1b():
    pw1 = 0.1;
    pw2 = 1 - pw1;
    plt.plot(ax, gx(mu1, sigma, pw1))
    plt.plot(ax, gx(mu2, math.sqrt(2) * sigma, pw2))
    plt.show()
    return

def q1c():
    dataset10 = createRandomDataset(10, prior2)
    dataset100 = createRandomDataset(100, prior2)
    #for dataPoint in dataset100:
        #print(str(dataPoint.X) + ' - ' + str(dataPoint.Y))
    return

def createRandomDataset(size, prior):
    arr = [0] * size
    i = 0
    for x in arr:
        dataPoint = Data()
        dataPoint.X = random.randint(MIN_RANGE, MAX_RANGE)
        dataPoint.Y = prior[dataPoint.X]
        arr[i] = dataPoint
        i = i + 1
    return arr

def q1d():
    pw1 = 0.25
    pw2 = pw1
    pw3 = 2 * pw1
    plt.plot(ax, gx(mu1, sigma, pw1))
    plt.plot(ax, gx(mu2, math.sqrt(2) * sigma, pw2))
    plt.plot(ax, gx(mu3, math.sqrt(3) * sigma, pw3))
    plt.show()
    return

if(len(sys.argv) == 2):
    arg = sys.argv[1]
    if arg == 'a':
        q1a()
    elif arg == 'b':
        q1b()
    elif arg == 'd':
        q1d()
else:
    q1d()
