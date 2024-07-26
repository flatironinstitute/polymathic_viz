import argparse
import os
import numpy as np
import polymathic_viz as p
import json
import importlib
import random
import time
import logging
from tqdm import tqdm
importlib.reload(p)

parser = argparse.ArgumentParser()
parser.add_argument("json_file", help="JSON file containing BIGGIF necessary info")
parser.add_argument("-s", "--save_raw", help="saves the intermediate individual gifs in UNPROCESSED and PROCESSED folder", action="store_true")
parser.add_argument("-i", "--info", help="Turn on logging to show live action", action="store_true")
args = parser.parse_args()

if args.info:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
else:
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

if args.json_file:
    def create_biggif_from_json(json_path, save_inter_res=False):
        # Read in the JSON file provided
        try:
            with open(json_path, 'r') as file:
                data = json.load(file)
        except FileNotFoundError:
            logging.error("The file was not found.")
            return
        except json.JSONDecodeError:
            logging.error("Error decoding JSON.")
            return
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return
        
        if save_inter_res:
            p.clean_folder(p.unprocessed_dir)
            p.clean_folder(p.processed_dir)
            
        start_time = time.time()
        numRows = data["numRows"]
        numCols = data["numColumns"]
        sw = data["squareImageWidth"]
        sh = data["squareImageHeight"]
        sl = data["standardTimeLength"]

        biggif = np.zeros((sl, numRows * sh, numCols * sw, 3), dtype=np.uint8)
        bh = biggif.shape[1]

        # this dictionary is used for gif placement support in each column
        # keys=column numbers, values=a dictionary of row : row_pixel_starting_point
        crdict = {}
        if sw % 2 != 0 or sh % 2 != 0:
            logging.warning("squareImageWidth and squareImageHeight must be even")
            return

        # initialize the crdict
        for spot in data["grid"]:
            col = spot["column"]
            row = spot["rowInColumn"]
            if col not in crdict:
                crdict[col] = {row: spot["shape"]}
            else:
                if row in crdict[col]:
                    logging.warning(f"Overlapping row numbers: ({col},{row})")
                    return
                crdict[col][row] = spot["shape"]  # Use parentheses instead of double square brackets
        
        # Sort each column's list based on the row value
        crdict = {k: crdict[k] for k in sorted(crdict)}
        
        # replace the second argument with the row pixel value
        for col in crdict:
            rowp = 0
            for r in crdict[col]:
                if rowp >= bh:
                    logging.warning(f"Row pixel starting point out of range: ({col}, {rowp})")
                    return
                if crdict[col][r] == "square":
                    crdict[col][r] = rowp
                    rowp += sh
                else:
                    crdict[col][r] = rowp
                    rowp += sh // 2      

        # for each file in the grid
        for spot in tqdm(data["grid"], desc="Processing GIFs"):
            gif = spot["gif"]
            array, names = p.read_file_as_array(dataset_name=gif["dataset"], 
                                                t=gif["tnum"], 
                                                c=gif["condition"],
                                                d1=gif["dimension"][0], d2=gif["dimension"][1],
                                                overwrite=False,
                                                filepath=gif["filepath"], 
                                                fieldname=gif["fieldname"], slice_num=gif["slice"])
            
            if save_inter_res:
                p.write_gif(array, names, save_to_dir=p.unprocessed_dir, fps=7, dump=True, filepath=gif["filepath"])
            
            array = p.postprocess_it(array, gif["dataset"], sh, sw, sl)

            if save_inter_res:
                p.write_gif(array, names, save_to_dir=p.processed_dir, fps=7, dump=True, filepath=gif["filepath"])

            ts, h, w, _ = array.shape
            cut = random.randint(0, ts-1) 
            shuffled_file = np.roll(array, shift=cut, axis=0)

            # put this gif into the correct spot
            col = spot["column"]
            row = spot["rowInColumn"]
            col_pixel = sw * col
            if row not in crdict[col]:
                logging.warning(f"Row pixel has not been processed for row {row}")
                return
            row_pixel = crdict[col][row]

            logging.info(f"PUT INTO BIGGIF AT ({col_pixel}, {row_pixel})")
            biggif[:, row_pixel : row_pixel + h, col_pixel : col_pixel + w, :] = shuffled_file.copy()

            # Clear array memory
            del array
            del shuffled_file  # Optionally clear shuffled_file as well

        
        print("Saving BIGGIF...")
        gifversion = os.path.basename(json_path.replace(".json", ".gif"))
        p.save_gif(biggif, f"{p.output_dir}/{gifversion}", fps=7)
        print("Saving MP4...")
        mpversion = os.path.basename(json_path.replace(".json", ".mp4"))
        p.save_video(biggif, f"{p.output_dir}/{mpversion}", fps=7)
        print("Saving NPY...")
        npyversion = os.path.basename(json_path.replace(".json", ""))
        np.save(f"{p.output_dir}/{npyversion}", biggif)
        end_time = time.time()
        duration = end_time - start_time
        # display(Image.fromarray(biggif[0]))
        print(f"Output Stored in: {p.output_dir} ")
        print(f"Process Completed! Time taken: {duration:.4f} seconds.")

    if args.save_raw:
        create_biggif_from_json(args.json_file, save_inter_res=True)
    else:
        create_biggif_from_json(args.json_file)