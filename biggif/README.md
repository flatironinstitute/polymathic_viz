# Polymathic Visualization Tool

To set up environment, in terminal run:
```
python -m venv .env
source .env/bin/activate
```

To install necessary packages, in terminal run:
```
pip install --upgrade pip
pip install -r requirements.txt
```

## Code Structure

```
.
├── .gitignore
├── .env
├── README.md
├── requirements.txt
├── config_polymathic
│   ├── README.md
│   ├── acoustic_scattering_discontinuous_2d
│   │   └── config.json
│   ├── acoustic_scattering_inclusions_2d
│   │   └── config.json
│   ├── acoustic_scattering_maze_2d
│   │   └── config.json
│   └── ...
├── input
│   ├── README.md
│   └── biggif.json
└── lib
    ├── polymathic_viz.py
    ├── get_collection.py
    └── biggif_from_json.py
```

## Create BIGGIF Collage from JSON

Create a biggif collage from the given input biggif JSON file:
```bash
python lib/biggif_from_json.py input/biggif.json
```
*Output will be stored in `./output`*

**Optional Arguments**

* `-i` or `--info` is a flag to turn on logging info in terminal
* `-g` or `--gif` to get a .gif output
* `-s` or `--save_raw` is a flag to save intermediate results (the individual gifs as seperate .gif, .mp4, .npy files)
    * raw gifs will be saved in folder `./unprocessed`
    * resized, edited, cropped biggif-ready gifs will be saved in folder `./processed`


## Build up Collection

Get a collection of gifs of all fields in all datasets (with a file limit of 3 per dataset):
```bash
python lib/get_collection.py --dataset all
```

For help on all the ways to build up the collection, just run:
```bash
python lib/get_collection.py
```

## Help:

Before running any command, make sure you have read access to:
```
"/mnt/home/polymathic/ceph/the_well/datasets/"
```

If `biggif.mp4` files are not viewable in browsers, run:
```
ffmpeg -i final3.mp4 -vcodec libx264 -acodec aac biggif.mp4
```

