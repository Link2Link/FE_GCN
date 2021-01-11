import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--file', type=str,
                        default='/home/llx/work_dir/output/pointpillar/default/eval/eval_all_default/default/log_eval_20210107-103817.txt',
                        help='specify the file')
    parser.add_argument('--obj', type=int,
                        default=0,
                        help='specify the obj, car:0 ped:1 cyc:2')

    args = parser.parse_args()
    return args

def pick_from_one_file(file, save=True, obj=0):
    with open(file, "r") as f:
        data = f.read()

    pattern = re.compile(r'\*+ EPOCH 71 EVALUATION \*+')
    start = re.search(pattern, data).span()[0]

    data = data[start:]

    car_list = []
    ped_list = []
    cyc_list = []

    for i in range(71, 81):
        re_pattern = r'\*+ EPOCH {epoch} EVALUATION \*+'.format(epoch=i)
        pattern_start = re.compile(re_pattern)
        re_pattern = r'{epoch} has been evaluated'.format(epoch=i)
        pattern_end = re.compile(re_pattern)
        batch_start = re.search(pattern_start, data).span()[0]
        batch_data = data[batch_start:]

        batch_end = re.search(pattern_end, batch_data).span()[0]
        batch_data = batch_data[:batch_end]

        pattern = re.compile(r'3d   AP:\d+\.\d+\, \d+\.\d+\, \d+\.\d+')
        string = pattern.findall(batch_data)
        car_ped_cyc = [string[0][8:], string[4][8:], string[8][8:]]

        car = list(map(lambda s: float(s), car_ped_cyc[0].split(',')))
        ped = list(map(lambda s: float(s), car_ped_cyc[1].split(',')))
        cyc = list(map(lambda s: float(s), car_ped_cyc[2].split(',')))

        car_list.append(car)
        ped_list.append(ped)
        cyc_list.append(cyc)

    car_npy = np.array(car_list)
    ped_npy = np.array(ped_list)
    cyc_npy = np.array(cyc_list)

    npy = np.concatenate([car_npy, ped_npy, cyc_npy], axis=1)
    best_epoch = np.argmax(npy[:, 1+obj*3])
    best_eval = npy[best_epoch][np.newaxis, :]

    npy = np.concatenate([npy, best_eval], axis=0)

    df = pd.DataFrame(npy)
    df.rename(columns={0: 'car:easy',
                       1:'car:moderate',
                       2: 'car:hard',
                       3: 'ped:easy',
                       4: 'ped:moderate',
                       5: 'ped:hard',
                       6: 'cyc:easy',
                       7: 'cyc:moderate',
                       8: 'cyc:hard',
                       }, inplace=True)
    obj_dict = {0:'car', 1:'ped', 2:'cyc'}
    df.rename(index={0: 71,
                     1: 72,
                     2: 73,
                     3: 74,
                     4: 75,
                     5: 76,
                     6: 77,
                     7: 78,
                     8: 79,
                     9: 80,
                     10: 'best '+ obj_dict[obj]}, inplace=True)

    if save:
        file_name = 'eval_pick.csv'
        df.to_csv(file.parent/file_name)

        pd.set_option('display.max_columns', None)
        print(df)
        print('save to ', file.parent/file_name)


if __name__ == '__main__':
    args = parse_config()
    file = Path(args.file)
    pick_from_one_file(file, obj=args.obj)

