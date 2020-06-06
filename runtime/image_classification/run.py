import argparse
import os

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-m', dest='models', type=str, nargs='+',
                    help='models', required=True)
parser.add_argument('-b', dest='batches', type=int, nargs='+',
                    help='num batches', required=True)
parser.add_argument('-c', dest='configs', type=str, nargs='+',
                    help='configs', default=['4_straight'])
parser.add_argument('-j', dest='jsons', type=str, nargs='+',
                    help='jsons and gpus', default=['mp_config'])

args = parser.parse_args()
args = vars(args)

print("\n ----------------- \n Arguments\n", args)

for m in args["models"]:
    for c in args["configs"]:
        for j in args["jsons"]:
            for b in args["batches"]:
                ranks = 1 if '1_conf' in j else c.split('_')[0]
                run_args = f" {m} {c} {j} {ranks} {b}"
                print(f"Running {run_args}")
                os.system(f"./run.sh {run_args}")
