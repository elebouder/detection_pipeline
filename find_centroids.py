import os
import csv


def get_centroid(l):
    xmin = float(l[0])
    ymin = float(l[1])
    xmax = float(l[2])
    ymax = float(l[3])

    cx = (xmin + xmax)/2
    cy = (ymin + ymax)/2
    return cx, cy


csvfile = '/home/elebouder/Data/landsat/detection_csv/2015_5.csv'


with open(csvfile,'r') as csvinput:
    with open(csvfile, 'w') as csvoutput:
        writer = csv.writer(csvoutput)
        reader = csv.reader(csvinput)

        all = []
        row = next(reader)
        row.append('c_x')
        row.append('c_y')
        all.append(row)

        for row in reader:
            cx, cy = get_centroid(row)
            row.append(cx)
            row.append(cy)
            all.append(row)

        writer.writerows(all)


