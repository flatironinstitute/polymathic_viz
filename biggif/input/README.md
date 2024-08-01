# JSON File describing BIGGIF

## JSON format

```
{
    "numRows": 4,
    "numColumns": 6,
    "squareImageWidth": 256,
    "squareImageHeight": 256,
    "standardTimeLength": 200,
    "grid": [
        {
            "shape": "square",
            "column": 0,
            "rowInColumn": 0,
            "gif": {
                "filepath": "/mnt/home/polymathic/ceph/the_well/datasets/active_matter/data/train/active_matter_L_10.0_zeta_1.0_alpha_-1.0.hdf5",
                "dataset": "active_matter",
                "tnum": 2,
                "fieldname": "E",
                "dimension": [
                    0,
                    0
                ],
                "condition": 0,
                "slice": null
            }
        },
        {
            "shape": "rectangle",
            "column": 0,
            "rowInColumn": 1,
            "gif": {
                "filepath": "/mnt/home/polymathic/ceph/the_well/datasets/convective_envelope_rsg/data/train/convective_envelope_rsg_trajectories_10.hdf5",
                "dataset": "convective_envelope_rsg",
                "tnum": 0,
                "fieldname": "density",
                "dimension": [
                    null,
                    null
                ],
                "condition": 0,
                "slice": 1
            }
        },
        ...
```


*  `numRows` and `numColumns` describes biggif shape
*  `squareImageWidth`, `squareImageHeight`, and `standardTimeLength` is used to create the biggif, and also used by postprocessing to resize all square gifs into these dimensions, and all rectangular gif height to `squareImageHeight // 2` 
*  `grid` represents the collage grid, with each spot having a `shape`, `column`, and `rowInColumn`
    * Use the `column` to change the column the gif is in
    * Use the `rowInColumn` to change the row the gif is in (space is allocated based on `shape`, overlapping or going over is not allowed)
* Each spot in the `grid` stores a `gif`, change fields in the gif to change the file stored at this spot
