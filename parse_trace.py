#!/usr/bin/python
import csv
import sys


def find_rows(kernel_name, kernel_name_list):
	rows = []
	i = 0
	for name in kernel_name_list:
		if (kernel_name in name):
			rows.append(i)
		i = i + 1;
	return rows

trace_file = str(sys.argv[1]);
mode = int(sys.argv[2]);

#mode: 0 total time, 1 kernel sum time, 2, kernel time

print("input trace:" + trace_file);


file = open(trace_file)
csv_reader = csv.reader(file)
time_stamp = []
kernel_time = []
kernel_name = []
skip=3
i = 0
for row in csv_reader:
	if (i > skip):
		time_stamp.append(row[0])
		kernel_time.append(row[1])
		kernel_name.append(row[18])
	i = i + 1;


# print(time_stamp)
# print(kernel_time)
# print(kernel_name)

if (mode == 0):
	first_kernel = str(sys.argv[3]);
	last_kernel = str(sys.argv[4]);

	start = find_rows(first_kernel, kernel_name)
	stop = find_rows(last_kernel, kernel_name)

	# print(start, stop)
	print(float(time_stamp[stop[-1]])-float(time_stamp[start[0]]))

if (mode == 1):
	kernel = str(sys.argv[3]);
	rows = find_rows(kernel, kernel_name)
	sum = 0.0
	for i in rows:
		sum += float(kernel_time[int(i)])
	print(sum)

if (mode == 2):
	kernel = str(sys.argv[3]);
	rows = find_rows(kernel, kernel_name)
	sum = 0.0
	idx = 1
	for i in rows:
		print(idx, float(kernel_time[int(i)]))
		idx = idx + 1

if (mode == 3):
	kernel = str(sys.argv[3]);
	size = int(sys.argv[4]);
	rows = find_rows(kernel, kernel_name)
	sum = 0.0
	idx = 1
	min = 10000
	min_idx = 0
	for i in rows:
		config = (7-idx)%7;
		time = float(kernel_time[int(i)])
		if (size >= config and time < min): 
			min = time
			min_idx = config
		# print(config, float(kernel_time[int(i)]))
		if (idx%7 == 0):
			print("min: " + str(min_idx) + " time: " + str(min))
			min = 10000
			size = size - 1;

		idx = idx + 1
		


