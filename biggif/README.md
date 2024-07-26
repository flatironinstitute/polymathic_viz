# Polymathic Visualization Tool

To set up environment, in terminal run:
```
python -m venv ./venv
source ./venv/bin/activate
```

To install necessary packages, in terminal run:
```
pip install -r requirements.txt
```

## Create BIGGIF Collage from JSON

**The ultimate command:**
```
python3 src/biggif_from_json.py ./jsonFiles/biggif.json -s -i
```

* `-s` or `--save_raw` is a flag to save intermediate results (the individual gifs as seperate .gif, .mp4, .npy files)
* `-i` or `--info` is a flag to turn on verbose mode

## Notes

Everytime you run the command, **three important folders** will be created: 

* `output` folder will store the results (always)
* `processed` folder will store the cropped/resized individual gifs (if `-s` flag is given)
* `unprocessed` folder will store the raw/unprocessed indidual gifs (if `-s` flag is given)
* (`collection` folder is used for a separate command)

## Help:

Make sure you have read access to:
```
"/mnt/home/polymathic/ceph/the_well/datasets/"
```

If `biggif.mp4` files are not viewable in browsers, run:
```
ffmpeg -i final3.mp4 -vcodec libx264 -acodec aac biggif.mp4
```

