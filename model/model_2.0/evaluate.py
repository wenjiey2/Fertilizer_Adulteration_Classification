import csv
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


def main():
    predictions = {}
    with open("predictions.csv", newline='\n') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            predictions[row[0]] = int(row[1])

    # print(predictions)

    real = [{}, {}, {}, {}, {}]
    types = ['clumped', 'clumped_maize', 'discolored', 'maize', 'normal']

    for i in range(len(types)):
        adulterated = 'adulterated/' + types[i] + '/*.jpg'
        clean = 'clean/' + types[i] + '/*.jpg'
        for file in glob.glob(adulterated) + glob.glob(adulterated.replace('jpg', 'JPG')):
            filename = file[file.rfind('/') + 1: file.rfind('.')]
            real[i][filename] = 0
        for file in glob.glob(clean) + glob.glob(clean.replace('jpg', 'JPG')):
            filename = file[file.rfind('/') + 1: file.rfind('.')]
            real[i][filename] = 1

    percentages = np.zeros([6, 12])
    for i in range(12):
        print('\nWhen threshold is %d: ' %(i+1), file=open("summary.txt", "a"))
        misclassified = 0
        for image_type in range(len(types)):
            a, b, c, d = 0, 0, 0, 0
            for file in real[image_type]:
                if real[image_type][file] == 0 and predictions[file] <= i:
                    a += 1
                elif real[image_type][file] == 0 and predictions[file] > i:
                    b += 1
                elif real[image_type][file] == 1 and predictions[file] <= i:
                    c += 1
                else:
                    d += 1
            percentages[image_type][i] = round((a + d) / (a + b + c + d), 3)
            percentages[5][i] += a + d
            print('%s confusion matrix: %d, %d, %d, %d' % (types[image_type], a, b, c, d), file=open("summary.txt", "a"))
            misclassified += (b + c)
        percentages[5][i] /= 1566
        print('%d images misclassified' % misclassified, file=open("summary.txt", "a"))

    # for accuracy in percentages:
    #     print(accuracy)

    x = np.arange(12)
    plt.plot(x+1, percentages[0])
    plt.plot(x+1, percentages[1])
    plt.plot(x+1, percentages[2])
    plt.plot(x+1, percentages[3])
    plt.plot(x+1, percentages[4])
    plt.plot(x+1, percentages[5])
    ax=plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    types.append('Average')
    plt.legend(types, loc='best')
    plt.savefig('Percentages.png')

    opt_threshold = np.zeros(6)
    for i in range(6):
        opt_threshold[i] = np.argmax(percentages[i])
    opt_av_threshold = int(opt_threshold[5])
    print("\nMax average accuracy: ", percentages[5][opt_av_threshold], file=open("summary.txt", "a"))
    print("Optimal threshold:", opt_av_threshold + 1, file=open("summary.txt", "a"))
    for i in range(5):
        print(types[i], "accuracy for the optimal threshold: ", percentages[i][opt_av_threshold], file=open("summary.txt", "a"))

if __name__ == "__main__":
    main()
