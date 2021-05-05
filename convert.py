#! /usr/bin/python
import numpy as np
import sys

in_file = sys.argv[1]
in_x = int(sys.argv[2])
in_y = int(sys.argv[3])
in_z = int(sys.argv[4])
in_prec = sys.argv[5]

out_file = sys.argv[6]
out_x = int(sys.argv[7])
out_y = int(sys.argv[8])
out_z = int(sys.argv[9])
out_prec = sys.argv[10]

print("input file: " + in_file) 
print("input size: {}*{}*{}".format(in_x, in_y, in_z))
in_type = ""
if (in_prec == "s"):
  print("input prec: single")
  in_type = np.float32
if (in_prec == "d"):
  print("input prec: double")
  in_type = np.float64

print("output file: " + out_file) 
print("output size: {}*{}*{}".format(out_x, out_y, out_z))
out_type = ""
if (out_prec == "s"):
  print("output prec: single")
  out_type = np.float32
if (out_prec == "d"):
  print("output prec: double")
  out_type = np.float64


in_array = np.fromfile(in_file, dtype = in_type)
if (in_array.shape[0] != in_x*in_y*in_z):
  print("Input size wrong!")
else:
  print("Input size correct.")
in_array = in_array.reshape((in_z, in_y, in_x))
out_array = np.zeros((out_z, out_y, out_x), dtype = out_type)

copy_z = min(in_z, out_z)
copy_y = min(in_y, out_y)
copy_x = min(in_x, out_x)

out_array[0:copy_z, 0:copy_y, 0:copy_x] = in_array[0:copy_z, 0:copy_y, 0:copy_x]
out_array.tofile(out_file)
