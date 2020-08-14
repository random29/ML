import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import random


def single_x(x):
    return x


def quad_x(x):
    return x * x


def triple_x(x):
    return x * x * x


def safe_ln(x, minval=0.0000000001):
    return np.log(x.clip(min=minval))


wLearn = np.zeros((92, 4))
amountOfSteps = 92
amountOfNumbers = 1000
x = np.linspace(0, 2 * np.pi, amountOfNumbers)
random.shuffle(x)
learn = x[:800]
tempVal = x[200:]
valid = tempVal[:100]
testEq = tempVal[100:]
a = np.array([np.sin, np.cos, safe_ln, np.exp, np.sqrt, single_x, quad_x,
              triple_x])
gt = 100 * np.sin(learn) + 0.5 * np.exp(learn) + 300
data = gt
# fig, axes = plt.subplots(2,2)
# axes[0][0].plot(data, "ro", markersize=2)

matrix_F = np.array([np.ones(len(learn)), np.sin(learn)]).transpose()
w = np.dot(np.dot(np.linalg.inv(np.dot(matrix_F.transpose(), matrix_F)), matrix_F.transpose()), data)
wLearn[0][0] = w[0]
wLearn[0][1] = w[1]
result = np.dot(w, matrix_F.transpose())
sum_temp = 0
count = 0
sumLearn = np.zeros(amountOfSteps)
while (count < len(learn)):
    sum_temp = sum_temp + (data[count] - result[count]) * (data[count] - result[count])
    count = count + 1
# print(sum_temp)
gt = 100 * np.sin(valid) + 0.5 * np.exp(valid) + 300
data = gt
sumLearn[0] = sum_temp
matrix_Valid = np.array([np.ones(len(valid)), np.sin(valid)]).transpose()
resultValid = np.dot(w, matrix_Valid.transpose())
sumValid = np.zeros(amountOfSteps)
sum_temp = 0
count = 0
while (count < len(valid)):
    sum_temp = sum_temp + (data[count] - resultValid[count]) * (data[count] - resultValid[count])
    count = count + 1
sumValid[0] = sum_temp
print(sum_temp)
universalCount = 1
# axes[0][0].plot(result, "bo", markersize=2)
# plt.show()
# arr[0][0] = np.sin.__name__
while (universalCount < 8):
    gt = 100 * np.sin(learn) + 0.5 * np.exp(learn) + 300
    data = gt
    matrix_F = np.array([np.ones(len(learn)), a[universalCount](learn)]).transpose()
    w = np.dot(np.dot(np.linalg.inv(np.dot(matrix_F.transpose(), matrix_F)), matrix_F.transpose()), data)
    wLearn[universalCount][0] = w[0]
    wLearn[universalCount][1] = w[1]
    result = np.dot(w, matrix_F.transpose())
    sum_temp = 0
    count = 0
    while (count < len(learn)):
        sum_temp = sum_temp + (data[count] - result[count]) * (data[count] - result[count])
        count = count + 1
    # print(sum_temp)
    gt = 100 * np.sin(valid) + 0.5 * np.exp(valid) + 300
    data = gt
    sumLearn[universalCount] = sum_temp
    matrix_Valid = np.array([np.ones(len(valid)), a[universalCount](valid)]).transpose()
    resultValid = np.dot(w, matrix_Valid.transpose())
    sum_temp = 0
    count = 0
    while (count < len(valid)):
        sum_temp = sum_temp + (data[count] - resultValid[count]) * (data[count] - resultValid[count])
        count = count + 1
    sumValid[universalCount] = sum_temp
    print(sum_temp)
    universalCount = universalCount + 1

combLearn = combinations(a, 2)
for pair in combLearn:
    gt = 100 * np.sin(learn) + 0.5 * np.exp(learn) + 300
    data = gt
    matrix_F = np.array([np.ones(len(learn)), pair[0](learn), pair[1](learn)]).transpose()
    w = np.dot(np.dot(np.linalg.inv(np.dot(matrix_F.transpose(), matrix_F)), matrix_F.transpose()), data)
    wLearn[universalCount][0] = w[0]
    wLearn[universalCount][1] = w[1]
    wLearn[universalCount][2] = w[2]
    result = np.dot(w, matrix_F.transpose())
    sum_temp = 0
    count = 0
    while (count < len(learn)):
        sum_temp = sum_temp + (data[count] - result[count]) * (data[count] - result[count])
        count = count + 1
    # print(sum_temp)
    gt = 100 * np.sin(valid) + 0.5 * np.exp(valid) + 300
    data = gt
    sumLearn[universalCount] = sum_temp
    matrix_Valid = np.array([np.ones(len(valid)), pair[0](valid), pair[1](valid)]).transpose()
    resultValid = np.dot(w, matrix_Valid.transpose())
    sum_temp = 0
    count = 0
    while (count < len(valid)):
        sum_temp = sum_temp + (data[count] - resultValid[count]) * (data[count] - resultValid[count])
        count = count + 1
    sumValid[universalCount] = sum_temp
    print(sum_temp)
    universalCount = universalCount + 1

combLearn3x = combinations(a, 3)

for pair in combLearn3x:
    gt = 100 * np.sin(learn) + 0.5 * np.exp(learn) + 300
    data = gt
    matrix_F = np.array([np.ones(len(learn)), pair[0](learn), pair[1](learn), pair[2](learn)]).transpose()
    w = np.dot(np.dot(np.linalg.inv(np.dot(matrix_F.transpose(), matrix_F)), matrix_F.transpose()), data)
    wLearn[universalCount] = w
    result = np.dot(w, matrix_F.transpose())
    sum_temp = 0
    count = 0
    while (count < len(learn)):
        sum_temp = sum_temp + (data[count] - result[count]) * (data[count] - result[count])
        count = count + 1
    # print(sum_temp)
    gt = 100 * np.sin(valid) + 0.5 * np.exp(valid) + 300
    data = gt
    sumLearn[universalCount] = sum_temp
    matrix_Valid = np.array([np.ones(len(valid)), pair[0](valid), pair[1](valid), pair[2](valid)]).transpose()
    resultValid = np.dot(w, matrix_Valid.transpose())
    sum_temp = 0
    count = 0
    while (count < len(valid)):
        sum_temp = sum_temp + (data[count] - resultValid[count]) * (data[count] - resultValid[count])
        count = count + 1
    sumValid[universalCount] = sum_temp
    print(sum_temp)
    universalCount = universalCount + 1

great = np.array(sumValid.argsort()[:3])
print()
# print(great)
wow = 0
contidences = np.ones(3)
contidencesValid = np.ones(3)
first = 'spam'
second = 'spam'
third = 'spam'
for i in great:
    # print(i)
    if (i < 8):
        print(a[i].__name__, sumValid[i], sumLearn[i], wLearn[i])
        test = str(wLearn[i][0]) + ' + ' + str(round(wLearn[i][1],1)) + ' * ' + a[i].__name__
        if (wow == 0):
            first = test
        elif (wow == 1):
            second = test
        elif (wow == 2):
            third = test
        contidences[wow] = sumLearn[i]
        contidencesValid[wow] = sumValid[i]
        wow = wow + 1
    elif (i < 36):
        combLearn = combinations(a, 2)
        i = i - 8
        temporary = 0
        for pair in combLearn:
            if (temporary == i):
                print(pair[0].__name__, pair[1].__name__, sumValid[i + 8], sumLearn[i + 8], wLearn[i + 8])
                test = str(round(wLearn[i + 8][0],1)) + ' + ' + str(round(wLearn[i + 8][1],1)) + ' * ' + pair[0].__name__ + ' + ' + str(round(
                    wLearn[i + 8][2],1)) + ' * ' + pair[1].__name__
                # print(test)
                if (wow == 0):
                    first = test
                elif (wow == 1):
                    second = test
                elif (wow == 2):
                    third = test
                contidences[wow] = sumLearn[i + 8]
                contidencesValid[wow] = sumValid[i + 8]
                wow = wow + 1
            temporary = temporary + 1
    elif (i < 92):
        combLearn3x = combinations(a, 3)
        i = i - 36
        temporary = 0
        for pair in combLearn3x:
            if (temporary == i):
                print(pair[0].__name__, pair[1].__name__, pair[2].__name__, sumValid[i + 36], sumLearn[i + 36],
                      wLearn[i + 36])
                test = str(round(wLearn[i + 36][0],1)) + ' + ' + str(round(wLearn[i + 36][1],1)) + ' * ' + pair[0].__name__ + ' + ' + str(round(
                    wLearn[i + 36][2],1)) + ' * ' + pair[
                           1].__name__ + ' + ' + str(round(wLearn[i + 36][3],1)) + ' * ' + pair[2].__name__
                if (wow == 0):
                    first = test
                elif (wow == 1):
                    second = test
                elif (wow == 2):
                    third = test
                contidences[wow] = sumLearn[i + 36]
                contidencesValid[wow] = sumValid[i + 36]
                wow = wow + 1
            temporary = temporary + 1

width = 0.3
# print(first, second, third)
labels = [first, second, third]
bin_poses = np.arange(len(labels))
bins_art = plt.bar(bin_poses, contidences, width, label="Обучающая")
plt.ylabel("Точность")
plt.title("Точнсть архитектора")
plt.xticks(bin_poses, labels)  # наши выборки
plt.legend(loc=3)
for rect in bins_art:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, rect.get_height() * 1.01,
             f"{height}",
             ha="center",
             va="bottom")

bin_poses = np.arange(len(labels))
bins_art = plt.bar(bin_poses + 0.3, contidencesValid, width, label="Valid")
plt.ylabel("Точность")
plt.title("Точнсть архитектора")
plt.xticks(bin_poses, labels)  # наши выборки
plt.legend(loc=3)
for rect in bins_art:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, rect.get_height() * 1.01,
             f"{height}",
             ha="center",
             va="bottom")
plt.show()