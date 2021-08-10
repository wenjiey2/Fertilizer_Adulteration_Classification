import cv2
import glob
import os


def main():
    folders = ['clumped', 'clumped_maize', 'discolored', 'maize', 'normal', 'pile']
    for folder in folders:
        foldername = 'adulterated\\' + folder + '\\*'
        for filename in glob.glob(foldername) + glob.glob('**\\' + foldername):
            preprocess(filename, 'preprocessed\\adulterated')
        foldername = 'clean\\' + folder + '\\*'
        for filename in glob.glob(foldername) + glob.glob('**\\' + foldername):
            preprocess(filename, 'preprocessed\\clean')
        print('Finished preprocessing ' + folder)


def preprocess(filename, output):
    if(not os.path.isdir(output)):
        os.makedirs(os.getcwd() + '\\' + output)
    file = filename[filename.rfind('\\') + 1: filename.rfind('.')]
    image = cv2.imread(filename)
    if image.shape[1] < image.shape[0]:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    image = cv2.resize(image, (1280, 960))
    pieces = []
    for row in range(3):
        for col in range(4):
            pieces.append(image[320 * row: 320 * (row + 1), 320 * col: 320 * (col + 1)])
    for i in range(len(pieces)):
        directory = output + '\\' + file + '_' + str(i) + '.jpg'
        cv2.imwrite(directory, pieces[i])


if __name__ == "__main__":
    main()
