from matplotlib.colors import LogNorm
from IPython.display import display
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import logging
import shutil
import random
import glob
import h5py
import time
import json
import cv2
import os
import re

def log_divider():
    if os.isatty(1):  # Check if stdout is a terminal
        terminal_width = os.get_terminal_size().columns
        divider = "-" * (terminal_width-33)
        logging.info(divider)
    else:
        logging.info("-------------------------------------------------------------")
    
def print_divider():
    if os.isatty(1):  # Check if stdout is a terminal
        terminal_width = os.get_terminal_size().columns
        divider = "-" * terminal_width
        print(divider)
    else:
        print("-------------------------------------------------------------")
    

script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of this script
base_dir = os.path.dirname(script_dir)  # Go up one level to src directory
config_dir = os.path.join(base_dir, 'config_polymathic')  
collection_dir = os.path.join(base_dir, 'collection')
unprocessed_dir = os.path.join(base_dir, 'unprocessed')
processed_dir = os.path.join(base_dir, 'processed')
output_dir = os.path.join(base_dir, 'output')

if not os.path.exists(collection_dir):
    os.makedirs(collection_dir)
    
if not os.path.exists(unprocessed_dir):
    os.makedirs(unprocessed_dir)
    
if not os.path.exists(processed_dir):
    os.makedirs(processed_dir)
    
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

datasets = [
    'the_well/datasets/acoustic_scattering_maze_2d',
    'the_well/datasets/active_matter',
    'the_well/datasets/euler_quadrants',
    'the_well/datasets/helmholtz_staircase',
    'the_well/datasets/pattern_formation',
    'the_well/datasets/planetswe',
    'the_well/datasets/rayleigh_benard',
    'the_well/datasets/shear_flow',
    'the_well/datasets/turbulent_radiative_layer_2D',
    'the_well/datasets/viscoelastic_instability',
    'the_well/datasets/convective_envelope_rsg',
    'the_well/datasets/MHD_256',
    'the_well/datasets/post_neutron_star_merger',
    'the_well/datasets/rayleigh_taylor_instability',
    'the_well/datasets/supernova_explosion_128',
    'the_well/datasets/turbulence_gravity_cooling',
    'the_well/datasets/turbulent_radiative_layer_3D'
]        


# datasets_set = {
#     'acoustic_scattering_maze_2d',
#     'active_matter',
#     'euler_quadrants',
#     'helmholtz_staircase',
#     'pattern_formation',
#     'planetswe',
#     'rayleigh_benard',
#     'shear_flow',
#     'turbulent_radiative_layer_2D',
#     'viscoelastic_instability',
#     'convective_envelope_rsg',
#     'MHD_256',
#     'post_neutron_star_merger',
#     'rayleigh_taylor_instability',
#     'supernova_explosion_128',
#     'turbulence_gravity_cooling',
#     'turbulent_radiative_layer_3D'
# }

def print_datasets():
    print("dataset_num \tdataset_name (dimension)")
    print_divider()
    for i, link in enumerate(datasets):
        dataset_name = os.path.basename(link)
        config_data = get_dataset_config(dataset_name, config_dir)
        if config_data is None:
            return
        is2D = config_data.get('is2D')
        shape = "2D" if is2D else "3D"
        print(f"\t{i}\t{dataset_name} ({shape})")
        
        
def get_dataset_config(dataset_name, config_dir=config_dir):
    for _, dirs, _ in os.walk(config_dir):
        if dataset_name in dirs:
            dataset_path = os.path.join(config_dir, dataset_name)
            config_file_path = os.path.join(dataset_path, 'config.json')
            if os.path.exists(config_file_path):
                with open(config_file_path, 'r') as config_file:
                    config_data = json.load(config_file)
                return config_data
    logging.info("Didn't find config_data.")
    logging.info("Make sure this script is under src/")
    logging.info("Make sure config_polymathic is at the same level as src/")
    return None

def clean_folder(dir):
    if os.path.exists(dir) and os.path.isdir(dir):
        # Remove everything in the directory
        shutil.rmtree(dir)
        # Optionally recreate the empty directory
        os.makedirs(dir)
        print(f"All contents of {dir} have been removed.")
    else:
        print(f"{dir} does not exist or is not a directory.")
    

def read_file_as_array(dataset_num=None, dataset_name=None,
                       f=0, t=0, k=-1, c=0, 
                       config_dir=config_dir,
                       color=None, d1=0, d2=0,
                       stop_at_filename=False,
                       overwrite=True,
                       filepath=None, fieldname=None, slice_num=None):
    if dataset_name:
        for i, link in enumerate(datasets):
            if dataset_name in link: 
                dataset_num = i
                break
    if dataset_num is None:
        logging.info("Read Docs for Help")
        return None, None
    # Edge Cases
    if dataset_num >= len(datasets) or dataset_num < 0:
        logging.warning("Invalid dataset number")
        return None, None

    if slice_num is None:
        slice_num = -1
    
    start_time = time.time()
    log_divider()
    dataset_name = os.path.basename(datasets[dataset_num])
    config_data = get_dataset_config(dataset_name, config_dir)
    paths = config_data.get('paths')
    if filepath:
        if slice_num >= 0: 
            logging.info(f"Reading {os.path.basename(filepath)} slice {slice_num} as numpy array...")
        else: 
            logging.info(f"Reading {os.path.basename(filepath)} as numpy array...")
    else:
        logging.info(f"Reading {dataset_name} file {f if f >= 0 else len(paths) + f}/{len(paths) - 1} as numpy array...")
    logging.info(f"Dataset Name ({dataset_num}/{len(datasets) - 1}): \t \t{dataset_name}")
    if config_data is None:
        return None, None
    is2D = config_data.get('is2D')
    
    if not is2D and (t==1 or t==2):
        logging.info("Visualizing t1 or t2 for 3D datasets is not supported.")
        return None, None
    
    # Special Checks
    if overwrite:
        if dataset_name == "euler_quadrants" and k == -1: 
            logging.info("(Overwrite) Euler_quadrants first field selected")
            # k = 0
            if k != 0:
                return None, None
        
        if dataset_name == "acoustic_scattering_maze_2d":
            logging.info("(Overwrite) Acoustic Scattering Maze last t0_field selected")
            if k != 2 or t != 0:
                return None, None
            
        if dataset_name == "rayleigh_taylor_instability":
            logging.info("(Overwrite) rayleigh_taylor_instability At_75 density selected")
            f = 4
            if t != 0 or k != 0:
                return None, None
    
        if dataset_name == "post_neutron_star_merger":
            logging.info("(Overwrite) post_neutron_star_merger electron_fraction selected")
            if t != 0 or k != 1:
                return None, None
            
        if dataset_name == "planetswe":
            logging.info("(Overwrite) planetswe only want t1 field")
            if t != 1:
                return None, None
        
    # Processing
    if stop_at_filename:
        if f >= len(paths):
            logging.warning(f"File number {f if f > 0 else len(paths) + f} out of range, there is only {len(paths)} files")
            return None, None
        if filepath:
            filename = os.path.basename(filepath)
        else:
            filename = os.path.basename(paths[f])
        logging.info(f"File (f={f if f > 0 else len(paths) + f}/{len(paths) - 1}): \t \t{filename} ")
        return None, None
        
    if not is2D: 
        array, names = process_3d(dataset_name, 
                                  config_data, 
                                  f, t, k, c,
                                  color, filepath, fieldname, slice_num)
    else: 
        array, names = process_2d(dataset_name, 
                                  config_data, 
                                  f, t, k, c,
                                  color, d1, d2, filepath, fieldname)

    # convert to RGB from RGBA for consistency after converting to gif
    if array is None: 
        end_time = time.time()
        duration = end_time - start_time
        logging.info(f"Process Completed! Time taken: {duration:.4f} seconds.")
        log_divider()
        return None, None
        
    array = array[..., :3]
    logging.info(f"Data Shape: \t \t \t{array.shape}")
    end_time = time.time()
    duration = end_time - start_time
    logging.info(f"Process Completed! Time taken: {duration:.4f} seconds.")
    log_divider()
    return array, names


def extract_info(config_data, f, t, k, color, filepath, fieldname):
    paths = config_data.get('paths')
    
    if f >= len(paths):
        logging.warning(f"File number {f if f > 0 else len(paths) + f} out of range, there is only {len(paths)} files")
        paths = None
    else:
        if filepath:
            filename = os.path.basename(filepath)
        else:
            filename = os.path.basename(paths[f])
        logging.info(f"File (f={f if f > 0 else len(paths) + f}/{len(paths) - 1}): \t \t{filename} ")

        color = config_data.get('color') if not color else color
        colormap = plt.get_cmap(color)
        logging.info(f"Colormap: \t \t \t{color}")

        scale = config_data.get('scale')
        logging.info(f"Scale: \t \t \t{scale}")

        if t == 2:
            keys = config_data.get('t2')
        elif t == 1:
            keys = config_data.get('t1')
        else:
            keys = config_data.get('t0')

        if len(keys) == 0:
            logging.warning(f"t{t}_fields list is empty")
            key = None
        else:
            logging.info(f"Keys (t={t}/2): \t \t \tt{t}_fields - {keys}")

            if fieldname:
                key = fieldname
                k = -1
                for i, name in enumerate(keys):
                    if name == key:
                        k = i
                        break
                if k == -1:
                    logging.error(f"key {fieldname} not found in t{t} Field")
                logging.info(f"t{t} Field (k={k if k != -1 else len(keys) - 1}/{len(keys) - 1}): \t \t{key}")
            elif k < len(keys):
                key = keys[k]
                logging.info(f"t{t} Field (k={k if k != -1 else len(keys) - 1}/{len(keys) - 1}): \t \t{key}")
            else:
                logging.warning(f"field number {k} is out of range")
                key = None

    return paths, colormap, scale, key


def preprocess_data_2d(file, t, c, key, d1, d2):
    if t == 2:
        readin = file['t2_fields'][key] # (3, 81, 256, 256, 2, 2)

        if len(readin.shape) == 6:
            iv, ts, h, w, _, _ = readin.shape

            data = np.empty((ts, h, w), dtype = readin.dtype)
            for i in range(ts):
                data[i] = readin[c, i, :, :, d1, d2]

            logging.info(f"Dimension (d1, d2): \t \tDimension ({d1}, {d2})")
        else:
            logging.warning(f"Sorry, Don't know how to handle this shape: {data.shape}")
            return None

    elif t == 1:
        readin = file['t1_fields'][key] # (3, 81, 512, 512, 2) 

        if len(readin.shape) == 5:
            iv, ts, h, w, _ = readin.shape

            data = np.empty((ts, h, w), dtype = readin.dtype)
            for i in range(ts):
                data[i] = readin[c, i, :, :, d1]

            logging.info(f"Dimension (d1): \t \tDimension {d1}")

        else:
            logging.warning(f"Sorry, Don't know how to handle this shape: {data.shape}")
            return None
    else:
        data = file['t0_fields'][key]   

        if len(data.shape) > 3:
            iv = data.shape[0]
            data = data[c]
        else:
            iv = 1
    logging.info(f"Initial Condition (c={c}/{iv - 1}): \tCondition number {c}")

    if len(data.shape) <= 2:
        logging.warning(f"This field ({key}) is not a visualizable gif")
        return None

    if not isinstance(data, np.ndarray):
        logging.warning(f"This field ({key}) is not supposed to be visualized")
        return None
    
    return data


# helper function for read_file_as_array
def process_2d(dataset_name, config_data, f, t, k, c, color, d1, d2, filepath, fieldname):
    logging.info(f"Dataset Dimensions: \t \t2D")
    
    paths, colormap, scale, key = extract_info(config_data, f, t, k, color, filepath, fieldname)
    if paths is None or key is None:
        return None, None
    
    names = []
    if filepath:
        file_to_open = filepath
    else:
        file_to_open = paths[f]
    with h5py.File(file_to_open, 'r') as file:
        # dataset specific preprocessing
        if dataset_name == "acoustic_scattering_maze_2d":
            mask = get_mask_acoustic(file)
        data = preprocess_data_2d(file, t, c, key, d1, d2)
        if dataset_name == "planetswe":
            data = data[:300]
        
        if data is None:
            return None, None
        
        ts, h, w = data.shape
        array = np.empty((ts, h, w, 4), dtype=np.uint8)
        
        # for active matter, we don't want to normalize across all timestamps
        # for asm2d, don't normalize for now
        if dataset_name != "active_matter" and dataset_name != "acoustic_scattering_maze_2d":
            if scale == 'normal':
                norm = plt.Normalize(vmin=data.min(), vmax=data.max())
                data = norm(data)
            elif scale == 'lognorm':
                norm = LogNorm(vmin=data.min(), vmax=data.max())
                data = norm(data)
        
        # fill in each timestamp in array
        if dataset_name == "active_matter":
            for i in range(ts):
                norm = plt.Normalize(vmin=data[i].min(), vmax=data[i].max())
                array[i] = colormap(norm(data[i]), bytes=True)
        else:
            for i in range(ts):
                array[i] = colormap(data[i], bytes=True)
                
        if dataset_name == "acoustic_scattering_maze_2d":
            for i in range(ts):
                array[i][mask == 0] = 255
    
    if t == 2:
        name = f"{dataset_name}-f{f if f > 0 else len(paths) + f}-t{t}-{key}-d({d1},{d2})-c{c}"
    elif t == 1:
        name = f"{dataset_name}-f{f if f > 0 else len(paths) + f}-t{t}-{key}-d{d1}-c{c}"
    else:
        name = f"{dataset_name}-f{f if f > 0 else len(paths) + f}-t{t}-{key}-c{c}"
    names.append(name)    
    return array, names


def get_mask_acoustic(file):
    density = file['t0_fields']['density']
    density = np.flip(density[0], axis=(0, 1))
    h, w = density.shape
    mask = np.ones((h, w), dtype=int)
    mask[density == density[0][0]] = 0
    return mask


def process_3d_new(dataset_name, config_data, f, t, k, c, color, filepath, fieldname, slice_num):
    logging.info(f"Dataset Dimensions: \t \t3D")
    
    paths, colormap, scale, key = extract_info(config_data, f, t, k, color, filepath, fieldname)
    if paths is None or key is None:
        return None, None
    
    names = []
    file_to_open = filepath if filepath else paths[f]
    
    with h5py.File(file_to_open, 'r') as file:
        data = file['t0_fields'][key]
        iv = data.shape[0]
        
        data = data[c]
        ts, d, h, w = data.shape
        logging.info(f"Initial Condition (c={c}/{iv - 1}): \tCondition number {c}")
        
        depth_fractions = [1/6, 1/3, 1/2, 2/3, 5/6]
        if slice_num >= 0: 
            depth_fractions = [depth_fractions[slice_num]]
        
        if slice_num >= 0:
            array_shape = (ts, d, h, 4) if dataset_name != "rayleigh_taylor_instability" else (ts, h, d, 4)
        else:
            array_shape = (5, ts, d, h, 4) if dataset_name != "rayleigh_taylor_instability" else (5, ts, h, d, 4)
        
        array = np.zeros(array_shape, dtype=np.uint8)
        
        for i, frac in enumerate(depth_fractions):
            if dataset_name in ["post_neutron_star_merger", "turbulent_radiative_layer_3D"]:
                s = int(w * frac)
                slice = data[:, :, :, s]
            else:
                s = int(d * frac)
                slice = data[:, s, :, :]
                if dataset_name == "rayleigh_taylor_instability":
                    slice = np.array([np.rot90(slice[j]) for j in range(ts)])
                    slice = slice[:-1]
                    ts -= 1

            if dataset_name in ["convective_envelope_rsg", "MHD_256"]:
                slice = np.array([plt.Normalize(vmin=slice[j].min(), vmax=slice[j].max())(slice[j]) for j in range(ts)])

            if scale == 'normal':
                norm = plt.Normalize(vmin=slice.min(), vmax=slice.max())
                slice = norm(slice)
            elif scale == 'lognorm':
                slice_flat = slice.flatten()
                norm = LogNorm()
                slice_normalized = norm(slice_flat)
                slice = slice_normalized.reshape(slice.shape)

            if slice_num >= 0:
                for j in range(ts):
                    array[j] = colormap(slice[j], bytes=True)
            else:
                for j in range(ts):
                    array[i][j] = colormap(slice[j], bytes=True)
                
            name = f"{dataset_name}-f{f if f > 0 else len(paths) + f}-{key}-s{slice_num if slice_num >= 0 else i}-c{c}"
            names.append(name)

    return array, names
    
def process_3d(dataset_name, config_data, f, t, k, c, color, filepath, fieldname, slice_num):
    logging.info(f"Dataset Dimensions: \t \t3D")
    
    paths, colormap, scale, key = extract_info(config_data, f, t, k, color, filepath, fieldname)
    if paths is None or key is None:
        return None, None
    
    names = []
    if filepath:
        file_to_open = filepath
    else:
        file_to_open = paths[f]
    with h5py.File(file_to_open, 'r') as file:
        data = file['t0_fields'][key]
        iv = data.shape[0]
        
        data = data[c]
        ts, d, h, w = data.shape
        logging.info(f"Initial Condition (c={c}/{iv - 1}): \tCondition number {c}")
        # initialize empty array
        depth_fractions = [1/6, 1/3, 1/2, 2/3, 5/6]
        if slice_num >= 0: 
            depth_fractions = [depth_fractions[slice_num]]
        array = None
        for i, frac in enumerate(depth_fractions):
            if dataset_name == "post_neutron_star_merger" or dataset_name == "turbulent_radiative_layer_3D":
                s = int(w * frac)
                slice = data[:, :, :, s]
                if array is None:
                    if slice_num >= 0:
                        array = np.zeros((ts, d, h, 4), dtype=np.uint8)
                    else:
                        array = np.zeros((5, ts, d, h, 4), dtype=np.uint8)
            else:
                s = int(d * frac)
                slice = data[:, s, :, :]
                if dataset_name == "rayleigh_taylor_instability":
                    for j in range(ts):
                        slice[j] = np.rot90(slice[j])
                    slice = slice[:-1]
                    ts -= 1
                    if array is None: 
                        if slice_num >= 0:
                            array = np.zeros((ts, d, h, 4), dtype=np.uint8)
                        else:
                            array = np.zeros((5, ts, d, h, 4), dtype=np.uint8)
                if array is None: 
                    if slice_num >= 0:
                        array = np.zeros((ts, h, d, 4), dtype=np.uint8)
                    else:
                        array = np.zeros((5, ts, h, d, 4), dtype=np.uint8)

            if dataset_name == "convective_envelope_rsg" or dataset_name == "MHD_256":
                for j in range(ts):
                    norm = plt.Normalize(vmin=slice[j].min(), vmax=slice[j].max())
                    slice[j] = norm(slice[j]) 

            if scale == 'normal':
                norm = plt.Normalize(vmin=slice.min(), vmax=slice.max())
                slice = norm(slice) 
            elif scale == 'lognorm':
                slice_flat = slice.flatten()
                norm = LogNorm()
                slice_normalized = norm(slice_flat)
                slice = slice_normalized.reshape(slice.shape)

            if slice_num >= 0:
                for j in range(ts):
                    array[j] = colormap(slice[j], bytes=True)
            else:
                for j in range(ts):
                    array[i][j] = colormap(slice[j], bytes=True)
                
            # gif name with s in it
            if slice_num >= 0: 
                name = f"{dataset_name}-f{f if f > 0 else len(paths) + f}-{key}-s{slice_num}-c{c}"
            else:
                name = f"{dataset_name}-f{f if f > 0 else len(paths) + f}-{key}-s{i}-c{c}"
            names.append(name)
    return array, names


def save_gif(frames, path, fps):
    images = [Image.fromarray(frame, 'RGB') for frame in frames]
    images[0].save(path, save_all=True, append_images=images[1:], 
                   optimize=False, duration=1000 // fps, loop=0)


def write_gif(array, names, save_to_dir=output_dir, fps=7, dump=False, filepath=None):
    start = time.time()
    
    if names == None or len(names) < 1:
        logging.warning("there is nothing to save")
        return
    
    dataset_name = names[0].split('-')[0]
    config_data = get_dataset_config(dataset_name, config_dir)
    if config_data is None:
        return
    match = re.search(r'f(\d+)', names[0])
    if match:
        f = int(match.group(1))
    else:
        logging.warning("No file number found in filename")
        return
    paths = config_data.get('paths')
    if filepath:
        file_to_open = filepath
    else:
        file_to_open = paths[f]
    filename = os.path.basename(file_to_open)
    if dump:
        save_to = save_to_dir
    else:
        save_to = f"{save_to_dir}/{dataset_name}/{filename}"
        
    if not os.path.exists(save_to):
        os.makedirs(save_to)

    if len(names) == 1:
        logging.info(f"Saving {len(names)} gif...")
        save_gif(array, f"{save_to}/{names[0]}.gif", fps)
        save_video(array,  f"{save_to}/{names[0]}.mp4", fps=fps)
        np.save(f'{save_to}/{names[0]}', array)
    else:
        logging.info(f"Saving {len(names)} gifs...")
        subfolder = re.sub(r's\d+-', '', names[0])
        save_to = f"{save_to}/{subfolder}"
        if not os.path.exists(save_to):
            os.makedirs(save_to)
        for i, name in enumerate(names):
            save_gif(array[i], f"{save_to}/{name}.gif", fps)
            save_video(array[i],  f"{save_to}/{names}.mp4", fps=fps)
            np.save(f'{save_to}/{name}', array[i])
    
    logging.info(f"fps={fps}")
    logging.info("Gif(s) saved as:")
    for name in names:
        logging.info(f"\t{save_to}/{name}.gif")

    end = time.time()
    duration = end - start
    logging.info(f"File saved! Time taken: \t{duration:.4f} seconds.")
    log_divider()

def copy_dataset(dataset_num, limit=None, source_dir=collection_dir, destination_dir=unprocessed_dir):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Initialize an empty dictionary to store the file paths
    dataset_name = os.path.basename(datasets[dataset_num])

    # Initialize the dictionary to store files
    unshuffled = {}

    # Walk through the directory structure
    for root, dirs, files in os.walk(source_dir):
        # Skip directories like .ipynb_checkpoints
        dirs[:] = [d for d in dirs if d != '.ipynb_checkpoints']
        
        for file in files:
            if dataset_name in file:
                if file.endswith('.gif'):
                    source_file_path = os.path.join(root, file)
                    destination_file_path = os.path.join(destination_dir, file)
                    if not os.path.exists(destination_file_path):
                        unshuffled[source_file_path] = destination_file_path
                    else:
                        logging.debug(f"{destination_file_path} already exists")
                        
    if limit:
        files_to_copy = {}
        # Randomly select up to `limit` pairs of gif and npy files
        shuffled_items = list(unshuffled.items())
        random.shuffle(shuffled_items)  # Shuffle items to randomize selection
        for source_file, destination_file in shuffled_items[:limit]:
            files_to_copy[source_file] = destination_file
            npy_source_file = source_file.replace(".gif", ".npy")
            npy_destination_file = destination_file.replace(".gif", ".npy")
            if os.path.exists(npy_source_file):
                files_to_copy[npy_source_file] = npy_destination_file
    else:
        files_to_copy = {}
        for source_file, destination_file in unshuffled.items():
            files_to_copy[source_file] = destination_file
            npy_source_file = source_file.replace(".gif", ".npy")
            npy_destination_file = destination_file.replace(".gif", ".npy")
            if os.path.exists(npy_source_file):
                files_to_copy[npy_source_file] = npy_destination_file

    # Copy the files
    for source_file, destination_file in files_to_copy.items():
        shutil.copy2(source_file, destination_file)
        logging.debug(f'Copied: {source_file} to {destination_file}')   

    logging.info(f"Copied {len(files_to_copy) // 2} new gifs from dataset {dataset_name}")

def copy_all(limit=None, source_dir=collection_dir, destination_dir=unprocessed_dir):
    for i in range(len(datasets)):
        copy_dataset(i, limit=limit)

def postprocess(dataset_num, from_dir=unprocessed_dir, to_dir=processed_dir, mode="wx", sh=256, sw=256, sl=200):
    files = glob.glob(f"{from_dir}/*.npy")
    d = os.path.basename(datasets[dataset_num])
    for i, f in enumerate(files):
        logging.info(f"Files left in 'unprocessed' folder: {len(files) - i}")
        logging.info(f"Reading file: {f}")
        dataset_name = os.path.basename(f).split('-')[0]
        if dataset_num and dataset_name != d:
                continue
        logging.info(f"Dataset name: {dataset_name}")
        gif_file_version = f.replace("npy", "gif")
        array = np.load(f)
        ts, h, w, _ = array.shape
        logging.info(f"The image is of shape: {h} x {w}")
        logging.info(f"Timestamps is : {ts}")
        if 'r' in mode: 
            logging.info("mode r: display image")
            display(Image.fromarray(array[ts // 2]))
            
        # special processing based on dataset
        if dataset_name == "turbulent_radiative_layer_2D":
            processed = array[:, :, :256, :]
            if 'r' in mode: 
                logging.info("mode r: display cropped version for turbulent_radiative_layer_2D")
                display(Image.fromarray(processed[ts // 2]))   
        else:
            # processing based on dimensions
            if h == w:
                if h != sh:
                    logging.info(f"Resizing the image to {sh} x {sw}")
                    if dataset_name == "turbulence_gravity_cooling":
                        processed = np.zeros((ts - 10, sh, sw, 3), dtype=array.dtype)
                        for t in range(ts - 10):
                            img = Image.fromarray(array[t + 10])
                            processed[t] = np.array(img.resize((sw, sh)))
                    else:
                        processed = np.zeros((ts, sh, sw, 3), dtype=array.dtype)
                        
                        for t in range(ts):
                            img = Image.fromarray(array[t])
                            processed[t] = np.array(img.resize((sw, sh)))
                        
                    if 'r' in mode: 
                        display(Image.fromarray(processed[0]))
                        display(Image.fromarray(processed[ts // 2]))
                        display(Image.fromarray(processed[ts - 10 - 1]))
                else:
                    processed = array
            elif h > w:
                processed = np.zeros((ts, sh // 2, sw, 3), dtype=array.dtype)
                for t in range(ts):
                    img = Image.fromarray(array[t])
                    if h != 512 or w != 128: 
                        slice = np.array(img.resize((128, 512)))
                    else: 
                        slice = array[t]
                    if dataset_name == "rayleigh_benard":
                        processed[t] = np.rot90(slice[128:384, :], k=-1)
                    else:
                        processed[t] = np.rot90(slice[128:384, :])
                if 'r' in mode: 
                    logging.info(f"Resized and rotated version for {dataset_name}")
                    display(Image.fromarray(processed[ts // 2]))
            else:
                if h == sh // 2 and w == sw:
                    processed = array
                else:
                    processed = np.zeros((ts, sh // 2, sw, 3), dtype=array.dtype)
                    for t in range(ts):
                        img = Image.fromarray(array[t])
                        processed[t] = np.array(img.resize((sw, sh // 2)))
                    if 'r' in mode: 
                        logging.info(f"Resized to {sh // 2} x {sw}  for {dataset_name}")
                        display(Image.fromarray(processed[ts // 2]))

        if 'w' in mode: 
            logging.info("mode w: write")
            # cut or expand the gif
            ts, h, w, _ = processed.shape
            if ts > sl:
                processed = processed[:sl, :, :, :]
            elif ts < sl:
                new_processed = np.zeros((sl, h, w, 3), dtype=processed.dtype)
                for t in range(sl):
                    new_processed[t] = processed[t % ts]
                processed = new_processed

            write_gif(processed, [os.path.basename(f).split('.')[0]], save_to_dir=to_dir, dump=True)
                
        if os.path.exists(f) and 'x' in mode:
            logging.info("mode x: remove")
            if os.path.exists(f):
                logging.info(f"Remove {f}")
                os.remove(f)
            if os.path.exists(gif_file_version):
                logging.info(f"Remove {gif_file_version}")
                os.remove(gif_file_version)
            
            
        # process one at a time
        logging.info(f"Success! Processed array shape: {processed.shape}")
        log_divider()

def postprocess_it(array, dataset_name, sh, sw, sl):
    logging.info(f"Making array BIGGIF-READY")
    logging.info(f"Dataset name: {dataset_name}")
    ts, h, w, _ = array.shape
    logging.info(f"The image is of shape: {h} x {w}")
    logging.info(f"Timestamps is : {ts}")
    
    # special processing based on dataset
    if dataset_name == "turbulent_radiative_layer_2D":
        processed = array[:, :, :256, :]
    else:
        # processing based on dimensions
        if h == w:
            if h != sh:
                logging.info(f"Resizing the image to {sh} x {sw}")
                if dataset_name == "turbulence_gravity_cooling":
                    processed = np.zeros((ts - 10, sh, sw, 3), dtype=array.dtype)
                    for t in range(ts - 10):
                        img = Image.fromarray(array[t + 10])
                        processed[t] = np.array(img.resize((sw, sh)))
                else:
                    processed = np.zeros((ts, sh, sw, 3), dtype=array.dtype)
                    
                    for t in range(ts):
                        img = Image.fromarray(array[t])
                        processed[t] = np.array(img.resize((sw, sh)))
            else:
                processed = array
        elif h > w:
            processed = np.zeros((ts, sh // 2, sw, 3), dtype=array.dtype)
            for t in range(ts):
                img = Image.fromarray(array[t])
                if h != 512 or w != 128: 
                    slice = np.array(img.resize((128, 512)))
                else: 
                    slice = array[t]
                if dataset_name == "rayleigh_benard":
                    processed[t] = np.rot90(slice[128:384, :], k=-1)
                else:
                    processed[t] = np.rot90(slice[128:384, :])
        else:
            if h == sh // 2 and w == sw:
                processed = array
            else:
                processed = np.zeros((ts, sh // 2, sw, 3), dtype=array.dtype)
                for t in range(ts):
                    img = Image.fromarray(array[t])
                    processed[t] = np.array(img.resize((sw, sh // 2)))

    # cut or expand the gif to sl
    ts, h, w, _ = processed.shape
    if ts > sl:
        processed = processed[:sl, :, :, :]
    elif ts < sl:
        new_processed = np.zeros((sl, h, w, 3), dtype=processed.dtype)
        for t in range(sl):
            new_processed[t] = processed[t % ts]
        processed = new_processed

    # process one at a time
    logging.info(f"Success! Processed array shape: {processed.shape}")
    log_divider()

    return processed
    
    
def postprocess_next(from_dir=unprocessed_dir, to_dir=processed_dir, mode="r", sh=256, sw=256, sl=200):
    files = glob.glob(f"{from_dir}/*.npy")
    logging.info(f"Files left: {len(files)}")
    for f in files:
        logging.info(f"Reading file: {f}")
        dataset_name = os.path.basename(f).split('-')[0]
        logging.info(f"Dataset name: {dataset_name}")
        gif_file_version = f.replace("npy", "gif")
        array = np.load(f)
        ts, h, w, _ = array.shape
        logging.info(f"The image is of shape: {h} x {w}")
        logging.info(f"Timestamps is : {ts}")
        if 'r' in mode: 
            logging.info("mode r: display image")
            display(Image.fromarray(array[ts // 2]))
            
        # special processing based on dataset
        if dataset_name == "turbulent_radiative_layer_2D":
            processed = array[:, :, :256, :]
            if 'r' in mode: 
                logging.info("mode r: display cropped version for turbulent_radiative_layer_2D")
                display(Image.fromarray(processed[ts // 2]))   
        else:
            # processing based on dimensions
            if h == w:
                if h != sh:
                    logging.info(f"Resizing the image to {sh} x {sw}")
                    if dataset_name == "turbulence_gravity_cooling":
                        processed = np.zeros((ts - 10, sh, sw, 3), dtype=array.dtype)
                        for t in range(ts - 10):
                            img = Image.fromarray(array[t + 10])
                            processed[t] = np.array(img.resize((sw, sh)))
                    else:
                        processed = np.zeros((ts, sh, sw, 3), dtype=array.dtype)
                        
                        for t in range(ts):
                            img = Image.fromarray(array[t])
                            processed[t] = np.array(img.resize((sw, sh)))
                        
                    if 'r' in mode: 
                        display(Image.fromarray(processed[0]))
                        display(Image.fromarray(processed[ts // 2]))
                        display(Image.fromarray(processed[ts - 10 - 1]))
                else:
                    processed = array
            elif h > w:
                processed = np.zeros((ts, sh // 2, sw, 3), dtype=array.dtype)
                for t in range(ts):
                    img = Image.fromarray(array[t])
                    if h != 512 or w != 128: 
                        slice = np.array(img.resize((128, 512)))
                    else: 
                        slice = array[t]
                    if dataset_name == "rayleigh_benard":
                        processed[t] = np.rot90(slice[128:384, :], k=-1)
                    else:
                        processed[t] = np.rot90(slice[128:384, :])
                if 'r' in mode: 
                    logging.info(f"Resized and rotated version for {dataset_name}")
                    display(Image.fromarray(processed[ts // 2]))
            else:
                if h == sh // 2 and w == sw:
                    processed = array
                else:
                    processed = np.zeros((ts, sh // 2, sw, 3), dtype=array.dtype)
                    for t in range(ts):
                        img = Image.fromarray(array[t])
                        processed[t] = np.array(img.resize((sw, sh // 2)))
                    if 'r' in mode: 
                        logging.info(f"Resized to {sh // 2} x {sw}  for {dataset_name}")
                        display(Image.fromarray(processed[ts // 2]))

        if 'w' in mode: 
            logging.info("mode w: write")
            # cut or expand the gif
            ts, h, w, _ = processed.shape
            if ts > sl:
                processed = processed[:sl, :, :, :]
            elif ts < sl:
                new_processed = np.zeros((sl, h, w, 3), dtype=processed.dtype)
                for t in range(sl):
                    new_processed[t] = processed[t % ts]
                processed = new_processed

            write_gif(processed, [os.path.basename(f).split('.')[0]], save_to_dir=to_dir, dump=True)
                
        if os.path.exists(f) and 'x' in mode:
            logging.info("mode x: remove")
            if os.path.exists(f):
                logging.info(f"Remove {f}")
                os.remove(f)
            if os.path.exists(gif_file_version):
                logging.info(f"Remove {gif_file_version}")
                os.remove(gif_file_version)
            
            
        # process one at a time
        logging.info(f"Success! Final array shape: {processed.shape}")
        break
            
def save_video(frames, path, fps):
    # Get the dimensions from the first frame
    height, width, _ = frames[0].shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(path, fourcc, fps, (width, height))

    for frame in frames:
        # Convert frame from RGB (PIL) to BGR (OpenCV)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(frame_bgr)

    video.release()  # Finalize the video file


def recreate(source, save_as, numRows, numCols, fps=7, compress=1, ih=256, iw=256, mode="d"):
    files = np.load(source)
    
    if len(files) < numRows * numCols:
        logging.info(f"Number of files {len(files)} is not enough to recreate a size {numRows * numCols} gif")
        return
    
    biggif = None
    nr, nc = 0, 0
    for f in files:
        # print(f)
        array = np.load(f)
        ts, h, w, _ = array.shape
        # print(h, w)

        # create the blank canvas
        if biggif is None: 
            if compress % 2 == 1:
                biggif = np.zeros((ts // compress, numRows * ih, numCols * iw, 3), dtype=array.dtype)
            else:
                biggif = np.zeros((ts // compress + 1, numRows * ih, numCols * iw, 3), dtype=array.dtype)
            bts, bh, _, _ = biggif.shape

        # shuffle this array
        cut = random.randint(0, ts-1) 
        shuffled_file = np.roll(array, shift=cut, axis=0)

        # put this gif into the correct spot
        biggif[:, nr : nr + h, nc : nc + w, :] = shuffled_file[::compress].copy()

        # Clear array memory
        del array
        del shuffled_file  # Optionally clear shuffled_file as well
        
        # update pointers
        if nr < bh - h:
            nr += h
        else:
            nc += w
            nr = 0
    logging.info(f"gif processed. shape: {biggif.shape}")
    if 'd' in mode:
        logging.info("displaying gif...")
        display(Image.fromarray(biggif[bts//2]))
    if 's' in mode:
        logging.info("saving gif...")
        save_video(biggif, save_as, fps=fps)
        np.save(save_as.replace(".mp4", ""), files)
        logging.info("saved")
    
    
class collage_maker:

    def __init__(self, numRows, numColumns, ih=256, iw=256, npy_dir=processed_dir):
        self.files = glob.glob(f"{npy_dir}/*.npy")
        if len(self.files) == 0: raise ValueError("need at least one npy file")
        self.numRows = numRows
        self.numColumns = numColumns
        self.ih = ih
        self.iw = iw
        self.squares = set()
        self.rectangulars = set()
        for f in range(len(self.files)):
            array = np.load(self.files[f])
            self.ts, h, w, _ = array.shape
            if h != w:
                self.rectangulars.add(f)
            else:
                self.squares.add(f)
        if len(self.rectangulars) // 2 + len(self.squares) < numRows * numColumns:
            raise ValueError(f"need at least {numRows * numColumns} files ready, currently only have {len(self.rectangulars) // 2 + len(self.squares)} files")
        self.biggif = np.zeros((self.ts, numRows * ih, numColumns * iw, 3), dtype=array.dtype)
        self.nr, self.nc = 0, 0
        self.array = None
        self.choice = 0
        self.f = None
        self.collage_sequence = []
        
    def show_next(self, shape="s"):
        if len(self.squares) == 0 and len(self.rectangulars) == 0:
            logging.warning("All available gifs have been put in.")
            return self
        if shape == "s":
            if len(self.squares) == 0:
                logging.warning("Theres no more square gifs left")
                self.array = None
                return self
            self.choice = random.choice(list(self.squares)) 
            
        else:
            if len(self.rectangulars) == 0:
                logging.warning("Theres no more rectangular gifs left")
                self.array = None
                return self
            self.choice = random.choice(list(self.rectangulars)) 
            
        self.f = self.files[self.choice]
        self.array = np.load(self.f)
        _, h, w, _ = self.array.shape
        bts, _, _, _ = self.biggif.shape
        
        # put this file into the biggif
        snapshot = self.biggif[bts // 2, :, :, :].copy()
        snapshot[self.nr : self.nr + h, self.nc : self.nc + w, :] = self.array[self.ts // 2]
        display(Image.fromarray(snapshot))
        return self

    def put(self):
        if self.array is None or self.f is None:
            return
        _, h, w, _ = self.array.shape
        _, bh, _, _ = self.biggif.shape
        # roll the array to make random start
        cut = random.randint(0, self.ts-1) 
        shuffled_file = np.roll(self.array, shift=cut, axis=0)

        # save the array in biggif
        for t in range(self.ts):
            self.biggif[t, self.nr : self.nr + h, self.nc : self.nc + w, :] = shuffled_file[t]
        self.collage_sequence.append(self.f)
        
        if self.nr < bh - h:
            self.nr += h
        else:
            self.nc += w
            self.nr = 0
        if h == w:
            self.squares.remove(self.choice)
        else:
            self.rectangulars.remove(self.choice)
            
    def save(self, to_dir=output_dir, save_as="final.mp4", fps=7):
        if not os.path.exists(to_dir):
            os.makedirs(to_dir)
        
        logging.info(f"Saving {to_dir}/{save_as}")
        save_video(self.biggif, f"{to_dir}/{save_as}.mp4", fps)
        
        # Escape double quotes in replace method
        logging.info(f"Saving collage sequence {to_dir}/{save_as.replace('.mp4', '.npy')}")
        np.save(f"{to_dir}/{save_as.replace('.mp4', '')}", self.collage_sequence)
        
        logging.info("Saved.")
