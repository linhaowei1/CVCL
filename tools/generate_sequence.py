import random

M_5T = ['M-5T-{}'.format(i) for i in range(5)]
C10_5T = ['C10-5T-{}'.format(i) for i in range(5)]
C100_20T = ['C100-20T-{}'.format(i) for i in range(20)]
C100_10T = ['C10-10T-{}'.format(i) for i in range(10)]
T_5T = ['T-5T-{}'.format(i) for i in range(5)]
T_10T = ['T-10T-{}'.format(i) for i in range(10)]

implemented_dataset = {
    'M_5T': M_5T, 'C100_10T':C100_10T, 'C10_5T': C10_5T, 'C100_20T': C100_20T, 'T_5T':T_5T, 'T_10T': T_10T
}

for dataset in implemented_dataset.keys():
    with open('./sequence/{}'.format(dataset), 'w') as f_random_seq:
        for repeat_num in range(10):
            f_random_seq.writelines('\t'.join(implemented_dataset[dataset]) + '\n')
            random.shuffle(implemented_dataset[dataset])



