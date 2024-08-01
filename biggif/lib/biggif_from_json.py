import polymathic_viz as p
import os
import importlib
import argparse
import logging

importlib.reload(p)

# CLI argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("json_file", help="JSON file containing BIGGIF necessary info")
parser.add_argument("-s", "--save_raw", help="saves the intermediate individual gifs in UNPROCESSED and PROCESSED folder", action="store_true")
parser.add_argument("-i", "--info", help="Turn on logging to show live action", action="store_true")
parser.add_argument("-g", "--gif", help="Output the result in .gif as well", action="store_true")
args = parser.parse_args()

# Set logging level based on flags passed in
if args.info: logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
else: logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# if a json file path is passed in
if args.json_file:
    if not os.path.exists(p.output_dir):
        os.makedirs(p.output_dir)
    
    print()
    p.print_divider()
    print("\t\tPOLYMATHIC VISUALIZATION TOOL")
    p.print_divider()
    print()

    w = False
    if args.gif:
        w = True
        
    s = False
    if args.save_raw:
        if not os.path.exists(p.unprocessed_dir):
            os.makedirs(p.unprocessed_dir)
            
        if not os.path.exists(p.processed_dir):
            os.makedirs(p.processed_dir)
        s = True
        
    p.create_biggif_from_json(args.json_file, save_inter_res=s, want_gif=w)