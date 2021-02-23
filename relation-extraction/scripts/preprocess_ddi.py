import sys

datapath = sys.argv[1]
datatype = sys.argv[2]

input_list = []

with open(file=datapath, mode='r') as f:
    for line in f.readlines():
        column = line.splitlines()[0].split('\t')[1:]
        input_list.append(column)

for idx, row in enumerate(input_list):
    if idx == 0 and datatype in ['train', 'dev']:
        print('sentence\tlabel')
        continue

    assert len(row) == len(input_list[0]), f"Error: Inconsistent number of columns elements. {len(row)} =/= {len(input_list[0])}"

    print('\t'.join(row))

    # if datatype == 'test':
    #     print('\t'.join(row[:2]))
    # else:
    #     print('\t'.join(row))
