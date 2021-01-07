import argparse
from pathlib import Path

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--file', type=str, default='/home/llx/work_dir/output/GCNpointpillar/PlainGCN/eval/eval_all_default/default/log_eval_20210107-060530.txt', help='specify the config for training')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_config()
    file = Path(args.file)
    print(file)

