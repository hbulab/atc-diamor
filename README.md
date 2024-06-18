# ATC - DIAMOR - Pedestrians

This repository provides code to work with the [DIAMOR and ATC data sets](https://dil.atr.jp/ISL/sets/groups/).

## Installation

### Requirements

Clone the repository and install the provided `pedestrians-social-binding` package with

```{bash}
git clone git@github.com:Chevrefeuille/atc-diamor-pedestrians.git
cd atc-diamor-pedestrian/code
pip install -e ./package
```

### Data (DIAMOR raw data)

The data sets are not provided with this repository. You can download the DIAMOR raw data and groups interaction annotations from the [DIAMOR and ATC data sets](https://dil.atr.jp/ISL/sets/groups/) website.

Create a `data` directory in the root of the repository and create three subdirectories `formatted`, `unformatted`, and `raw` in it.

In the `raw` directory, create a subdirectory `diamor` and place the DIAMOR raw trajectories (`.dat` files) in it (inside two subdirectories `06` and `08`).

In the `unformatted` directory, create a subdirectory `diamor` and create a subdirectory `annotations` in it. Place the DIAMOR groups annotations (`.pkl`and `.csv` files) in it.

The structure of the `data` directory should look like this:

```
data
├── formatted
├── raw
│   ├── diamor
│   │   ├── 06
│   │   │   ├── data_2_1.dat
│   │   │   ├── ...
│   │   ├── 08
│   │   │   ├── data_1_1.dat
│   │   │   ├── ...
└── unformatted
    └── diamor
        └── annotations
            ├── gt_2p_yoshioka_06.pkl
            ├── gt_2p_yoshioka_08.pkl
            ├── ids_wrt_group_size_taniguchi_06.pkl
            ├── ids_wrt_group_size_taniguchi_08.pkl
            ├── taniguchi_gt_gest_06.csv
         
```

To format the raw data and annotations, run the following command:

```{bash}
cd code/preprocessing
python format_diamor_raw.py
python prepare_diamor_granular_annotations.py
python format_diamor_annotations.py
```

### Examples

You can verify that the package was correctly installed and the data was correctly formatted by running

```{bash}
cd code
python 00_example_plot_trajectories.py
```
