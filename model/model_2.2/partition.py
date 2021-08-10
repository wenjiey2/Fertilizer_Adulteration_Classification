import cv2
import glob
import random
import os

def main():
    d = {}
    k = 6
    folder = 'fert_dataset\\fertilizer'

    for file in glob.glob(folder+'\\*.jpg'):
        filename = file[file.rfind('\\') + 1:]
        d[filename] = 0

    def extract(img):
        start = img.find('\\')
        return img[start + 1: ]

    keys = list(d.keys())
    keys = [extract(i) for i in keys]
    keys = list(dict.fromkeys(keys))
    random.seed(2020)
    random.shuffle(keys)

    each = int(len(keys) / k)
    partitions = [keys[i * each: (i + 1) * each] for i in range(k)]
    for i in range(k):
        for j in range(each):
            file = partitions[i][j]
            image = cv2.imread(folder+'\\' + file)
            for part in range(k):
                goto = 'ks\\k' + str(part)
                if(not os.path.isdir(goto)):
                    os.makedirs(os.getcwd() + '\\' + goto)
                if i == part:
                    goto += '\\test\\'
                else:
                    goto += '\\train\\'
                if d[file] == 0:
                    goto += folder+'\\'
                else:
                    goto += folder+'\\'
                if(not os.path.isdir(goto)):
                    os.makedirs(os.getcwd() + '\\' + goto)
                goto += file
                cv2.imwrite(goto, image)
        print("Finished part " + str(i))

if __name__ == "__main__":
    main()
