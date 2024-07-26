import argparse 
import os
import logging
import importlib
import polymathic_viz as p
import time
import glob

importlib.reload(p)

# Initialize parser object
parser = argparse.ArgumentParser()

parser.add_argument("-c", "--copy", help="copy gifs + npys from 'collection' to 'unprocessed'", action="store_true")
parser.add_argument("-p", "--process", help="processes all gifs from 'unprocessed' to 'processed'", action="store_true")
parser.add_argument("-d", "--dataset", help="select dataset numbers(s) to copy from", type=int, nargs='*')
parser.add_argument("-l", "--limit", help="limit the number of files copied/processed", type=int)
parser.add_argument("-cl", "--clean", help="removes all files from 'unprocessed' folder", action="store_true")
parser.add_argument("-db", "--debug", help="turn on logging debug and info", action="store_true")

# Get the arguments from the user
args = parser.parse_args()

# Handle the datasets argument
if args.dataset and len(args.dataset) == 0:
    p.print_datasets()

# Handle the verbose argument
if args.debug:
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
else:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    

# Handle the clean argument
if args.clean:
    print("removing ALL in 'unprocessed' folder")
    p.clean_unprocessed()

# Handle the copyall argument
if args.copy:
    if args.dataset:
        print(f"Copying from 'collection' to 'unprocessed' for dataset(s):")
        if args.limit:
            print(f"Limiting to {args.limit} files per dataset")
        for i in args.dataset:
            dataset = os.path.basename(p.datasets[i])
            print(f"\t{i}.\t{dataset}")
        for i in args.dataset:
            p.copy_dataset(i, limit=args.limit)
    else:
        print("Copying ALL from 'collection' to 'unprocessed' for all datasets")
        if args.limit:
            print(f"Limiting to {args.limit} files per dataset")
        p.copy_all(limit=args.limit)

# Handle the process argument
if args.process:
    if args.dataset:
        print(f"Processing gifs from 'unprocess' to 'processed' folder for dataset(s):")
        for i in args.dataset:
            dataset = os.path.basename(p.datasets[i])
            print(f"\t{i}.\t{dataset}")
        for i in args.dataset:
            p.postprocess(i)
        
    else:
        print(f"Processing all gifs from 'unprocessed' to 'processed' folder")
        for i in range(len(p.datasets)):
            p.postprocess(i)
    files = glob.glob(f"{p.unprocessed_dir}/*.npy")
    logging.info(f"Files left 'unprocessed': {len(files)}")
    files = glob.glob(f"{p.processed_dir}/*.npy")
    logging.info(f"Files in 'processed': {len(files)}")
    

