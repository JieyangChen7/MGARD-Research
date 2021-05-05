#!/usr/bin/python
import csv
import sys

filename = str(sys.argv[1])
nr = int(sys.argv[2])
nc = int(sys.argv[3])
nf = int(sys.argv[4])

r = int(sys.argv[5])
c = int(sys.argv[6])
f = int(sys.argv[7])


file = open(filename)
csv_reader = csv.reader(file)
results = []
for row in csv_reader:
	results.append(row)


print("({}, {}, {}: {})".format(r, c, f, results[r*nc+c][f]))


