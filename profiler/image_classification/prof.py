import argparse
import os

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-m', dest='models', type=str, nargs='+',
                    help='models', required=True)
parser.add_argument('-b', dest='batches', type=int, nargs='+',
                    help='num batches', required=True)
parser.add_argument('-d', dest='data', help='data dir', type=str, default='/N/u2/d/dnperera/data/imagenet-mini/')
parser.add_argument('-p', dest='profile output', help='profile ouptut dir', type=str, default='./profiles')

args = parser.parse_args()
args = vars(args)

print("\n ----------------- \n Arguments\n", args)

for m in args["models"]:
    for b in args["batches"]:
        run_args = f" -a {m}  -b {b} --data_dir {args['data']} --profile_directory {args['output']}/{b}"
        print(f"Running {run_args}")
        os.system(f"CUDA_VISIBLE_DEVICES=4 python main.py {run_args}")
