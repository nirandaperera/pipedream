import argparse
import os

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-m', dest='models', type=str, nargs='+',
                    help='models', required=True)
parser.add_argument('-b', dest='batches', type=int, nargs='+',
                    help='num batches', required=True)
parser.add_argument('-d', dest='data', type=str, default='/N/u2/d/dnperera/data/imagenet-mini/')
parser.add_argument('-p', dest='prof', help="profile dir", type=str,
                    default='../profiler/image_classification/profiles')
parser.add_argument('-c', dest='configs', type=str, nargs='+',
                    help='configs', default=['4_straight'])
parser.add_argument('-o', dest='out', help="output dir", type=str, default='./optim')
parser.add_argument('-n', dest='workers', help="workers", type=int, default=4)
parser.add_argument('-s', dest='size', help="mem size", type=int, default=11000000000)
parser.add_argument('--bw', dest='bw', help="bandwidth", type=int, default=2500000000)
parser.add_argument('--straight', action='store_true', help="use straight pipeline")

args = parser.parse_args()
args = vars(args)

print("\n ----------------- \n Arguments\n", args)

for m in args["models"]:
    for b in args["batches"]:
        run_args = f" -a {m}  -b {b} --data_dir {args['data']} --profile_directory {args['output']}/{b}"
        print(f"Running {run_args}")
        os.system(f"CUDA_VISIBLE_DEVICES=4 python main.py {run_args}")

for m in args["models"]:
    for c in args["configs"]:
        for b in args["batches"]:
            out_dir = f"{args['out']}/{b}/{m}/gpus={c}/"
            os.system(f"rm -rf {out_dir}/*")

            profile = f"{args['prof']}/{b}/{m}/graph.txt"

            straight = '--straight_pipeline' if args['straight'] else ''

            run_args = f" -f {profile} -n {args['workers']} -s {args['size']} -b {args['bw']} " \
                       f"-o {out_dir} --use_memory_constraint {straight}"
            print(f"Running {run_args}")
            os.system(f"python optimizer_graph_hierarchical.py {run_args}")

            run_args = f"-f {profile} -n {m} -a {m} -o {out_dir}"
