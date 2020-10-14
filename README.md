# imaging-biobuilders

Repository of imaging tools for the DTU BioBuilders of 2020.

## How to set up a python environment for running our code
In order to run our code, you need a working python environment. 

1. Download or clone the repository. 
2. Install the latest version of Miniconda for Python 3.8. https://docs.conda.io/en/latest/miniconda.html.
3. Open a terminal and navigate to the repository. 
4. Using the terminal, set up a virtual environment using `conda create -n env_name`. 
5. Activate environment using `activate env_name`
6. Install the packages required using `conda install --file requirements.txt`. Remember to navigate to the repository folder first. 

## How to run our Jupyter notebooks

1. If not already done, install Python and required packages using the steps above. 
2. Navigate to the repository on your computer. 
3. Activate the environment using `activate env_name`.
4. Boot up a notebook server by writing `jupyter notebook`. This should open a tab in your browser.
5. Click on a notebook to open it in a new tab. Cells of Python code can be run with Ctrl-Enter. 

## How to run parameter estimation and simulations on our data set with standard parameters. 

1. If not already done, install Python and required packages using the steps above. 
2. Navigate to the repository on your computer. 
3. Activate environment using `activate env_name`
4. Run the following command `python parameter_estimation_script.py --img_folder data/ --overview_file image_overview.csv --hours 24 --fluorescence_dim 2`
