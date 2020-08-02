import sys
import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt
sys.stdout = open('output.txt', 'wt')

# Input: observational interval tau, credible threshold pc, prior parameter alpha0, 
# positive feedback indicator u, robot UID id, period T, fill ratio (ratio of white tiles) f
tau =  [1, 5, 10, 15, 20, 25, 50, 75, 100, 150, 200, 250, 300]
pc = [0.9, 0.95, 0.98, 0.99]
alpha0 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 50]
u = [True, False]
T = 11
f = [0.52, 0.55, 0.6, 0.7, 0.8]
count = 0
one, zero = 0, 0
ones, zeros = [], []

# go through every possible combination of inputs
for k in f:
    for h in u:
        for j in pc:
            for m in tau:
                for n in alpha0:
                    count += 1
                    # white observation alpha, black obs beta, obs index i
                    alpha, beta = n, n
                    i = 0
                    # Output: binary classification of environment df
                    df = -1
                    for t in range(1, T):
                        #perform pseudorandom walk-->don't see relevance except in visual simulation
                        if t%m == 0:
                            #fill ratios: 0.52, 0.55, 0.6, 0.7, 0.8 (choices of 1)
                            C = np.random.choice([0, 1], p=[1-k, k])
                            alpha += C
                            beta += (1-C)
                            i += 1
                        
                        if df == -1:
                            p = ss.beta.cdf(0.5, alpha + n, beta + n, loc=0, scale=1)
                            if p > j:
                                df = 0
                            elif (1-p) > j:
                                df = 1

                        if df != -1 and u:
                            print("Case: %d, Positive feedback indicator: %s, Credible threshold: %.2f,"
                            " Fill ratio: %.2f, Observational interval: %d, Prior parameter: %d,"
                            " Observation Index: %d, Decision by probability: %d" % (count, h, j, k, m, n, i, df))
                        else:
                            print("Case: %d, Positive feedback indicator: %s, Credible threshold: %.2f,"
                            " Fill ratio: %.2f, Observational interval: %d, Prior parameter: %d,"
                            " Observation Index: %d, Decision by observation: %d" % (count, h, j, k, m, n, i, C))
                    if df == 1 or C == 1:
                        one += 1
                    elif df == 0 or C == 0:
                        zero += 1
    ones.append(one)
    zeros.append(zero)
    one, zero = 0, 0

# measure whether the fill ratio is decided as accurately as possible
decisionAccuracy = []
for i in range(5):
    decisionAccuracy.append(ones[i]/(ones[i]+zeros[i]))

# plots for each fill ratio and estimated accuracy
labels = ['0.52\n%.2f'%decisionAccuracy[0], '0.55\n%.2f'%decisionAccuracy[1], '0.60\n%.2f'%decisionAccuracy[2],\
         '0.70\n%.2f'%decisionAccuracy[3], '0.80\n%.2f'%decisionAccuracy[4]]
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, zeros, width, label='Zeros (Black tiles)')
rects2 = ax.bar(x + width/2, ones, width, label='Ones (White tiles)')


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Expected Fill ratios\nEstimated Accuracy')
ax.set_ylabel('Decisions')
ax.set_title('Decisions by fill ratios (percentage of white tiles)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

# I do not see the significance of this part unless it is to check the total number of experiments
# plt.figure()
# plt.bar(['Zeros (Black tiles)', 'Ones (White tiles)'], [sum(zeros), sum(ones)], align='center')
# plt.xlabel('Tile Colors')
# plt.ylabel('Total Decision Count')

plt.show()