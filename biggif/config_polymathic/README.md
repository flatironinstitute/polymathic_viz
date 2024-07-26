# Config Dir

This directory stores necessary configuration files for each dataset, that is necessary for processing.

```
{
    "dataset_name": the dataset's name,
    "num_hdf5_files": integer total number of hdf5 files,
    "is2D": true/false,
    "scale": normal/lognorm,
    "geometry": cartesian/spherical (not used for processing),
    "fps": fps to save this dataset in,
    "color": colormap,
    "t0": a list of available t0_fields,
    "t1": a list of available t1_fields,
    "t2": a list of available t2_fields,
    "file_names": a list of HDF5 file names,
    "paths": a list of paths to the actual HDF5 files
}

```