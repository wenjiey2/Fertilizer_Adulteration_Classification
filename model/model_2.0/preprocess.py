import cv2
import glob
import os


def main():
    folders = ['clumped', 'clumped_maize', 'discolored', 'maize', 'normal']
    for folder in folders:
        foldername = 'adulterated/' + folder + '/*.'
        for filename in glob.glob(foldername + 'jpg') + glob.glob(foldername + 'JPG'):
            preprocess(filename, 'preprocessed/adulterated')
        foldername = 'clean/' + folder + '/*.'
        for filename in glob.glob(foldername + 'jpg') + glob.glob(foldername + 'JPG'):
            preprocess(filename, 'preprocessed/clean')
        print('Finished preprocessing ' + folder)


def preprocess(filename, output):
    if(not os.path.isdir(output)):
        os.makedirs(os.getcwd() + '/' + output)
    file = filename[filename.rfind('/') + 1: filename.find('.')]
    print(file)
    image = cv2.imread(filename)
    if len(image) != 4032:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    image = cv2.resize(image, (400, 300))
    pieces = []
    for row in range(3):
        for col in range(4):
            pieces.append(image[100 * row: 100 * (row + 1), 100 * col: 100 * (col + 1)])
    for i in range(len(pieces)):
        directory = output + '/' + file + '_' + str(i) + '.jpg'
        cv2.imwrite(directory, pieces[i])


if __name__ == "__main__":
    main()
