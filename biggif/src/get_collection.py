import logging
import importlib
import os
import sys
import polymathic_viz as p
from polymathic_viz import print_divider, print_datasets, datasets, log_divider
import time

def print_help():
    print("Usage:")
    print("get_collection.py creates collection folder and fills it with gifs and raw numpy arrays for each dataset.\n")
    print_divider()
    print("Dataset Level Controls (required)")
    print_divider()
    print("Get list of all datasets with their dataset_num: \tpython3 src/get_collection.py -d")
    print("Process all datasets: \t \t \t \t \tpython3 src/get_collection.py -d all")
    print("Process a specific dataset: \t \t \t \tpython3 src/get_collection.py -d <int>")
    print()
    
    print_divider()
    print("File Level Controls (optional)")
    print_divider()
    print("Limit number of files to process: \t \t \tpython3 src/get_collection.py -lf <int>")
    print("Choose an HDF5 file number: \t \t \t \tpython3 src/get_collection.py -f <int>")
    print()
    
    print_divider()
    print("Field Level Controls (optional)")
    print_divider()
    print("Choose a t<>_field list number: \t \t \tpython3 src/get_collection.py -t <int>")
    print("Choose a specific field index within the list: \t \tpython3 src/get_collection.py -k <int>")
    print()

    print_divider()
    print("Initial Conditions Level Controls (optional)")
    print_divider()
    print("Choose a limit for number of conditions: \t \tpython3 src/get_collection.py -lc <int>")
    print("Choose a specifc initial condition number: \t \tpython3 src/get_collection.py -c <int>")
    print()

    print_divider()
    print("Other Useful flags: ")
    print_divider()
    print("[stop_at_file_name] - this breaks processing and prints out the filename of every file to be processed")
    print("[no_overwrite] - some fields/files were purposely skipped in each dataset because they were not desired, this turns that off and processes all")
    print()
    print()
    
    

def parse_arguments():     
    chosen_dataset = None
    limit_files = 3
    chosen_file = None
    chosen_t = [0, 1, 2]
    chosen_k = None
    limit_conditions = 1
    chosen_condition = 0
    stop_at_file_name = False
    overwrite = True

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "-d" or arg == "--dataset":
            if i + 1 < len(sys.argv):
                x = sys.argv[i + 1]
                if x.isdigit() and 0 <= int(x) < len(datasets):
                    chosen_dataset = int(x)
                    logging.info(f"Chosen dataset number {chosen_dataset}")
                elif x == "all":
                    logging.info(f"Processing all datasets...")
                    logging.info(f"This will take around 7 minutes to complete.")
                    i += 1
                    continue
                else:
                    logging.warning(f"{x} is not a valid dataset_num")
                    print_datasets() 
                    sys.exit()
            else:
                print_datasets() 
                sys.exit()
        elif arg == "-lf" and i + 1 < len(sys.argv) and sys.argv[i + 1].isdigit():
            limit_files = int(sys.argv[i + 1])
            logging.info(f"Number of HDF5 files limited to {limit_files}")
            i += 1
        elif arg == "-f" and i + 1 < len(sys.argv) and sys.argv[i + 1].isdigit():
            chosen_file = int(sys.argv[i + 1])
            logging.info(f"Chosen file number {chosen_file}")
            i += 1
        elif arg == "-t" and i + 1 < len(sys.argv) and sys.argv[i + 1].isdigit():
            t_val = int(sys.argv[i + 1])
            if 0 <= t_val <= 2:
                chosen_t = [t_val]
                logging.info(f"Chosen field number t{chosen_t}")
            else:
                logging.warning("Enter 0, 1, or 2 for t0_fields, t1_fields, or t2_fields")
                sys.exit()
            i += 1
        elif arg == "-k" and i + 1 < len(sys.argv) and sys.argv[i + 1].isdigit():
            chosen_k = int(sys.argv[i + 1])
            logging.info(f"Chosen field number {chosen_k}")
            i += 1
        elif arg == "-lc" and i + 1 < len(sys.argv) and sys.argv[i + 1].isdigit():
            limit_conditions = int(sys.argv[i + 1])
            logging.info(f"Limiting the first {limit_conditions} initial conditions")
            i += 1
        elif arg == "-c" and i + 1 < len(sys.argv) and sys.argv[i + 1].isdigit():
            chosen_condition = int(sys.argv[i + 1])
            logging.info(f"Chosen initial condition {chosen_condition}")
            i += 1
        elif arg == "stop_at_file_name":
            stop_at_file_name = True
        elif arg == "no_overwrite":
            overwrite = False
        i += 1

    return chosen_dataset, limit_files, chosen_file, chosen_t, chosen_k, limit_conditions, chosen_condition, stop_at_file_name, overwrite

def process_dataset(chosen_dataset, limit_files, chosen_file, chosen_t, chosen_k, limit_conditions, chosen_condition, stop_at_file_name, overwrite):
    dataset_range = [chosen_dataset] if chosen_dataset is not None else range(len(datasets))
    for n in dataset_range:
        dataset_name = os.path.basename(datasets[n])
        config_data = p.get_dataset_config(dataset_name)
        files_to_process = [chosen_file] if chosen_file is not None else range(-1, -limit_files - 1, -1)
        for i in files_to_process:
            for j in chosen_t:
                keys = config_data.get(f"t{j}")
                if chosen_k is None:
                    for ke in range(len(keys)):
                        process_file(n, i, j, ke, chosen_condition, limit_conditions, stop_at_file_name, overwrite)
                else:
                    process_file(n, i, j, chosen_k, chosen_condition, limit_conditions, stop_at_file_name, overwrite)

def process_file(dataset_index, file_index, t, k, chosen_condition, limit_conditions, stop_at_file_name, overwrite):
    for l in range(limit_conditions):
        if t == 2:
            for d1, d2 in [(0, 0), (0, 1)]:
                array, names = p.read_file_as_array(dataset_index, f=file_index, t=t, k=k, c=l if limit_conditions > 1 else chosen_condition, d1=d1, d2=d2, stop_at_filename=stop_at_file_name, overwrite=overwrite)
                p.write_gif(array, names, save_to_dir=p.collection_dir)
        else:
            array, names = p.read_file_as_array(dataset_index, f=file_index, t=t, k=k, c=l if limit_conditions > 1 else chosen_condition, stop_at_filename=stop_at_file_name, overwrite=overwrite)
            p.write_gif(array, names, save_to_dir=p.collection_dir)
            
def main():
    start_time = time.time()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    print()
    print_divider()
    print("\t\tPOLYMATHIC VISUALIZATION TOOL")
    print_divider()
    print()

    importlib.reload(p)

    if len(sys.argv) <= 1:
        print_help()
        return
        
    chosen_dataset, limit_files, chosen_file, chosen_t, chosen_k, limit_conditions, chosen_condition, stop_at_file_name, overwrite = parse_arguments()
    

    if limit_files == 3:
        logging.info("Default files limit: 3 (-lf <number of files> to overwrite)")
    if len(chosen_t) == 3:
        logging.info("Processing all available tfields")
    if not overwrite:
        logging.info("Overwriting turned off")

    log_divider()
    process_dataset(chosen_dataset, limit_files, chosen_file, chosen_t, chosen_k, limit_conditions, chosen_condition, stop_at_file_name, overwrite)

    end_time = time.time()
    duration = end_time - start_time
    logging.info(f"Main Completed! Time taken: {duration:.4f} seconds.")

if __name__ == "__main__":
    main()
