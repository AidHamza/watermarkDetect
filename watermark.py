# Copyright (C) 2013 Wesley Baugh
"""Detect watermark in images.
### Requires
- [Pillow](https://pypi.python.org/pypi/Pillow/2.0.0)
"""
import glob
import sys, getopt

from classify import MultinomialNB
from PIL import Image


TRAINING_POSITIVE = 'training-positive/*.jpg'
TRAINING_NEGATIVE = 'training-negative/*.jpg'

# Size of the Watermark image
RESIZED = (219, 40)


def get_image_data(infile):
    image = Image.open(infile)
    width, height = image.size
    
    thumb_width = 219
    thumb_height = 40

    left = (width - thumb_width)/2
    top = (height - thumb_height)/2
    right = (width + thumb_width)/2
    bottom = (height + thumb_height)/2

    # left upper right lower
    box = left, top, right, bottom
    print box
    region = image.crop(box)
    #region.save("test.jpeg")
    resized = region.resize(RESIZED)
    data = resized.getdata()
    # Convert RGB to simple averaged value.
    data = [sum(pixel) / 3 for pixel in data]
    # Combine location and value.
    values = []
    for location, value in enumerate(data):
        values.extend([location] * value)
    return values


def main(argv):
    # Get directory argument
    try:
        opts, args = getopt.getopt(argv,"h:d:",["help", "dir="])
    except getopt.GetoptError:
        print 'watermark.py -d <directory_name>'
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print 'watermark.py -d <directory_name>'
            sys.exit()
        elif opt in ("-d", "--dir"):
            directory = arg + "/*.jpg"

    try:
        if directory is None:
            print 'Directory flag empty, For help type: watermark.py -h'
            sys.exit(2)
    except NameError:
        print 'Directory flag not provided, For help type: watermark.py -h'
        sys.exit()


    print directory
    watermark = MultinomialNB()
    # Training
    count = 0
    for infile in glob.glob(TRAINING_POSITIVE):
        data = get_image_data(infile)
        watermark.train((data, 'positive'))
        count += 1
        print 'Training', count
    for infile in glob.glob(TRAINING_NEGATIVE):
        data = get_image_data(infile)
        watermark.train((data, 'negative'))
        count += 1
        print 'Training', count
    # Testing
    correct, total = 0, 0
    
    nonWatermarked = open("toWatermark.txt", "w")
    for infile in glob.glob(directory):
        data = get_image_data(infile)
        prediction = watermark.classify(data)
        if prediction.label == 'negative':
            nonWatermarked.write(infile + "\n")
            correct += 1
        total += 1
        print 'Testing ({0} / {1})'.format(correct, total)
    nonWatermarked.close()
    print 'Got', correct, 'out of', total, 'correct'


if __name__ == '__main__':
    main(sys.argv[1:])
