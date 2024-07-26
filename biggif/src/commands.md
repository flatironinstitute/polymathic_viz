# Less Used Commands

## Build up Collection

Get a collection of all fields in all datasets (with a file limit of 3 per dataset):
```
python3 src/get_collection.py --dataset all
```

## Postprocess to be BIGGIF-ready

Copy 1 file from each dataset in 'collection' to 'unprocessed' folder, and then process all files in 'unprocessed' folder to be "biggif ready" and move them to 'processed' folder
```
python3 src/postprocessor.py --copy --limit 1 --process
```

Instead of copying randomly, you can also copy paste the desired gifs/npy pairs into the unprocessed folder and then just run:
```
python3 src/postprocessor.py --process
```

Note: all processed gifs will have shape either (200, 256, 256, 3) or (200, 128, 256, 3), no distinction between 2D or 3D because only 2D slices are saved for 3D datasets.