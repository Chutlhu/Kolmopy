# Kolmopy


## Use custom dataset
You may want to add your own dataset. First you need to publish it as `.hdf5` file to speedup the dataloaders.

1. Create a new Python file associated to the new dataset, e.g., `myTurboDataset.py` in `./kolmopy/datasets/`
2. Implement and overwrite the methods of the class `core.Dataset`, in particular
    - `load_data()`
    - the properties to get all the variables and fields
    - `validate()`
3. Implement a method to convert the raw data in a hdf5 files:
    - `publish_as_hdf5()`
    - `main()` method to run it
4. Run the script as `python turb2D.py path/to/dataset/folder -o ./.cached/`