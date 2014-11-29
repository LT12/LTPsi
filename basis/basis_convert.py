#!/usr/bin/env python
import sys
import json
from collections import OrderedDict

def basis_convert(input_name,output_name):
	file_i = open(input_name,'r')
	lines = file_i.readlines()
	file_o = open(output_name,'w')
	file_o.write('basis_set = \\\n')
	basis_set = OrderedDict()
	for idx, line in enumerate(lines):
		if line.strip() == '****' and idx == 0:
			element = lines[idx + 1].split()[0]
			orbs = []
			elem_idx = idx + 1
		elif line.strip() == '****' and idx > 0:
			basis_set[element] = orbs
			if idx < len(lines) -1:
				element = lines[idx + 1].split()[0]
				orbs = []
				elem_idx = idx + 1
			
		if (any(line.split()[0] == i for i in ('S','P','D','F')) and
			idx != elem_idx):
			nG = int(line.split()[1])
			pG = [0] * nG
			for i in xrange(1,nG + 1):
				pG[i-1] = (float(lines[idx + i].split()[0]),
						   float(lines[idx + i].split()[1]))
			orbs.append((line.split()[0],pG))

		if (line.split()[0] == 'SP' and idx != elem_idx):
			nG = int(line.split()[1])
			pG = [0] * nG
			pG2 = [0] * nG
			for i in xrange(1,nG + 1):
				pG[i-1] = (float(lines[idx + i].split()[0]),
						   float(lines[idx + i].split()[1]))
				pG2[i-1] = (float(lines[idx + i].split()[0]),
						   float(lines[idx + i].split()[2]))
				
			orbs.append(('S', pG))
			orbs.append(('P', pG2))

	json.dump(basis_set,file_o,indent = 4)
	file_o.write('\n')
	file_o.write('\ndef getOrbs(atom):\n')
	file_o.write('    try:\n')
	file_o.write('        return basis_set[atom]\n')
	file_o.write('    except KeyError:\n')
	file_o.write("        raise NameError('Element not supported by basis set!')")
	file_i.close()
	file_o.close()

if __name__ == '__main__':
	input_file = sys.argv[1]
	output_file =  sys.argv[2]
	basis_convert(input_file,output_file)
