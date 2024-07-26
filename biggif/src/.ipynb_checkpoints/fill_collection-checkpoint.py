import logging
import importlib
import os
import sys
import polymathic_viz as p
from polymathic_viz import print_divider, print_datasets, datasets, log_divider
import time

def main():
    start_time = time.time()
    print()
    print_divider()
    print("\t \tPOLYMATHIC VISUALIZATION TOOL")
    print_divider()
    print()
    importlib.reload(p)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # process goes here
    # parse arguments
    
    # logging.info("Default processing: limit 3 HDF5 files from each dataset (to change specify -f <limit number>)")
    chosen_dataset = None
    limit_files = 3
    chosen_file = None
    chosen_t = [0, 1, 2]
    chosen_k = None
    limit_conditions = 1
    chosen_condition = 0
    stop_at_file_name = False
    overwrite=True
    
    # parse input arguments
    for i in range(1, len(sys.argv)):
        if sys.argv[i] == "-d":
            if i < len(sys.argv) - 1 and sys.argv[i + 1].isdigit():
                x = int(sys.argv[i+1])
                if x < 0 or x >= len(datasets):
                    logging.warning(f"{x} is not a valid dataset_num")
                    print_datasets()
                else:
                    # we want to process this dataset only
                    chosen_dataset = x
                    logging.info(f"Chosen dataset number {chosen_dataset}")  
            else:
                print_datasets()
                return
        elif sys.argv[i] == "-lf":
            if i < len(sys.argv) - 1 and sys.argv[i + 1].isdigit():
                limit_files = int(sys.argv[i+1]) 
                logging.info(f"Number of HDF5 files limited to {limit_files}")  
            else:
                logging.warning(f"Enter the Number of HDF5 files limit you want")          
                return
        elif sys.argv[i] == "-f":
            if i < len(sys.argv) - 1 and sys.argv[i + 1].isdigit():
                chosen_file = int(sys.argv[i+1]) 
                logging.info(f"Chosen file number {chosen_file}") 
            else:
                logging.warning("Enter the file number you want")
                return    
        elif sys.argv[i] == "-t":
            if i < len(sys.argv) - 1 and sys.argv[i + 1].isdigit() and int(sys.argv[i+1]) >= 0 and int(sys.argv[i+1]) <= 2:
                chosen_t = [int(sys.argv[i+1])] 
                logging.info(f"Chosen field number t{chosen_t}") 
            else:
                logging.warning(f"Enter 0, 1, or 2 for t0_fields, t1_fields, or t2_fields")
                return
        elif sys.argv[i] == "-k":
            if i < len(sys.argv) - 1 and sys.argv[i + 1].isdigit():
                chosen_k = int(sys.argv[i+1]) 
                logging.info(f"Chosen field number {chosen_k}")    
            else:
                logging.warning(f"Enter the field index you want")    
                return
        elif sys.argv[i] == "-lc":
            if i < len(sys.argv) - 1 and sys.argv[i + 1].isdigit():
                limit_conditions = int(sys.argv[i+1]) 
                logging.info(f"Limiting the first {limit_conditions} intial conditions")
            else:
                logging.warning(f"Enter the limit condition number you want")     
                return
        elif sys.argv[i] == "-c":
            if i < len(sys.argv) - 1 and sys.argv[i + 1].isdigit():
                chosen_condition = int(sys.argv[i+1]) 
                logging.info(f"Chosen initial condition {chosen_condition}")    
            else:
                logging.warning(f"Enter the intial condition number you want")  
                return
        elif sys.argv[i] == "stop_at_file_name":
            stop_at_file_name = True
        elif sys.argv[i] == "no_overwrite":
            overwrite = False
        else:
            pass
    
    
    if limit_files == 3:
        logging.info("Default files limit: 3 (-lf <number of files> to overwrite)")
        
    if len(chosen_t) == 3:
        logging.info("Processing all available tfields")
        
    if not overwrite:
        logging.info("Overwriting turned off")
    
        
    log_divider()
    if chosen_dataset is None:
        # looping back to front on all HDF5 files (assuming back is the newest)
        for n in range(len(datasets)):
            dataset_name= os.path.basename(datasets[n])
            config_data = p.get_dataset_config(dataset_name)
            for i in range(-1, limit_files * -1 - 1, -1):
                for j in chosen_t:
                    # if we have chosen a field to display
                    if chosen_k == None:
                        keys = config_data.get(f"t{j}")
                        for ke in range(len(keys)):
                            # if t2 fields
                            if j == 2:
                                if limit_conditions == 1:
                                    array, names = p.read_file_as_array(n, f=i, t=j, k=ke, c=chosen_condition, d1=0, d2=0, stop_at_filename=stop_at_file_name, overwrite=overwrite)
                                    array, names = p.read_file_as_array(n, f=i, t=j, k=ke, c=chosen_condition, d1=0, d2=1, stop_at_filename=stop_at_file_name, overwrite=overwrite)
                                else:
                                    for l in range(limit_conditions):
                                        array, names = p.read_file_as_array(n, f=i, t=j, k=ke, c=l,  d1=0, d2=0, stop_at_filename=stop_at_file_name, overwrite=overwrite)
                                        array, names = p.read_file_as_array(n, f=i, t=j, k=ke, c=l, d1=0, d2=1, stop_at_filename=stop_at_file_name, overwrite=overwrite)
                            else:
                                if limit_conditions == 1:
                                    array, names = p.read_file_as_array(n, f=i, t=j, k=ke, c=chosen_condition, stop_at_filename=stop_at_file_name, overwrite=overwrite)
                                else:
                                    for l in range(limit_conditions):
                                        array, names = p.read_file_as_array(n, f=i, t=j, k=ke, c=l, stop_at_filename=stop_at_file_name, overwrite=overwrite)
                    else:
                        # if t2 fields
                            if j == 2:
                                if limit_conditions == 1:
                                    array, names = p.read_file_as_array(n, f=i, t=j, k=chosen_k, c=chosen_condition, d1=0, d2=0, stop_at_filename=stop_at_file_name, overwrite=overwrite)
                                    array, names = p.read_file_as_array(n, f=i, t=j, k=chosen_k, c=chosen_condition, d1=0, d2=1, stop_at_filename=stop_at_file_name, overwrite=overwrite)
                                else:
                                    for l in range(limit_conditions):
                                        array, names = p.read_file_as_array(n, f=i, t=j, k=chosen_k, c=l,  d1=0, d2=0, stop_at_filename=stop_at_file_name, overwrite=overwrite)
                                        array, names = p.read_file_as_array(n, f=i, t=j, k=chosen_k, c=l, d1=0, d2=1, stop_at_filename=stop_at_file_name, overwrite=overwrite)
                            else:
                                if limit_conditions == 1:
                                    array, names = p.read_file_as_array(n, f=i, t=j, k=chosen_k, c=chosen_condition, stop_at_filename=stop_at_file_name, overwrite=overwrite)
                                else:
                                    for l in range(limit_conditions):
                                        array, names = p.read_file_as_array(n, f=i, t=j, k=chosen_k, c=l, stop_at_filename=stop_at_file_name, overwrite=overwrite)
    else:
        dataset_name= os.path.basename(datasets[chosen_dataset])
        config_data = p.get_dataset_config(dataset_name)
        # looping back to front on all HDF5 files (assuming back is the newest)
        for i in range(-1, limit_files * -1 - 1, -1):
            for j in chosen_t:
                if chosen_k == None:
                    keys = config_data.get(f"t{j}")
                    for ke in range(len(keys)):
                        if j == 2:
                            if limit_conditions == 1:
                                array, names = p.read_file_as_array(chosen_dataset, f=i, t=j, k=ke, c=chosen_condition, d1=0, d2=0, stop_at_filename=stop_at_file_name, overwrite=overwrite)
                                array, names = p.read_file_as_array(chosen_dataset, f=i, t=j, k=ke, c=chosen_condition, d1=1, d2=1, stop_at_filename=stop_at_file_name, overwrite=overwrite)
                            else:
                                for l in range(limit_conditions):
                                    array, names = p.read_file_as_array(chosen_dataset, f=i, t=j, k=ke, c=l, d1=0, d2=0, stop_at_filename=stop_at_file_name, overwrite=overwrite)
                                    array, names = p.read_file_as_array(chosen_dataset, f=i, t=j, k=ke, c=l, d1=0, d2=1, stop_at_filename=stop_at_file_name, overwrite=overwrite)
                        else:
                            if limit_conditions == 1:
                                array, names = p.read_file_as_array(chosen_dataset, f=i, t=j, k=ke, c=chosen_condition, stop_at_filename=stop_at_file_name, overwrite=overwrite)
                            else:
                                for l in range(limit_conditions):
                                    array, names = p.read_file_as_array(chosen_dataset, f=i, t=j, k=ke, c=l, stop_at_filename=stop_at_file_name, overwrite=overwrite)
                else:
                    if j == 2:
                        if limit_conditions == 1:
                            array, names = p.read_file_as_array(chosen_dataset, f=i, t=j, k=chosen_k, c=chosen_condition, d1=0, d2=0, stop_at_filename=stop_at_file_name, overwrite=overwrite)
                            array, names = p.read_file_as_array(chosen_dataset, f=i, t=j, k=chosen_k, c=chosen_condition, d1=1, d2=1, stop_at_filename=stop_at_file_name, overwrite=overwrite)
                        else:
                            for l in range(limit_conditions):
                                array, names = p.read_file_as_array(chosen_dataset, f=i, t=j, k=chosen_k, c=l, d1=0, d2=0, stop_at_filename=stop_at_file_name, overwrite=overwrite)
                                array, names = p.read_file_as_array(chosen_dataset, f=i, t=j, k=chosen_k, c=l, d1=0, d2=1, stop_at_filename=stop_at_file_name, overwrite=overwrite)
                    else:
                        if limit_conditions == 1:
                            array, names = p.read_file_as_array(chosen_dataset, f=i, t=j, k=chosen_k, c=chosen_condition, stop_at_filename=stop_at_file_name, overwrite=overwrite)
                        else:
                            for l in range(limit_conditions):
                                array, names = p.read_file_as_array(chosen_dataset, f=i, t=j, k=chosen_k, c=l, stop_at_filename=stop_at_file_name, overwrite=overwrite)
                    
        
    end_time = time.time()
    duration = end_time - start_time
    logging.info(f"Main Completed! Time taken: {duration:.4f} seconds.")
    
    
    
if __name__ == "__main__":
    main()