# Notes on Stats
This repository collects some statistical tools we might use during projects of computed vision based behavioral phenotyping.  

Durham 2020  
matias.di.martino.uy@gmail.com  

## Index:
Complete this once this is finished

## Repository tree.
```
├── LICENSE
├── README.md
├── enviroment.yml
├── doc
    ├── sections
        ├── secX.ipynb
        └── Sections files (md or notebooks)
    ├── figs
    └── tools
        ├── fileX.py 
        └── python source code. 
```

## License information

This code can be openly used for educational and research porpoises, any commercial use is forbidden without the explicit authorization of the authors.  

## Usage Instructions

- For installation, just create the virtual environment as instructed below.  
- To compile, run 
```bash
jb build doc/
```  
- After compilation, you will find the book in /_build/index.html

### Creating an Conda Environment

The conda environment is provided as `environment.yml`.  
1. `conda env create --file environment.yml`
2. `conda activate stats`

### Building a Jupyter Book

Run the following command in your terminal:
```bash
jb build doc/
```

If you would like to work with a clean build, you can empty the build folder by running:
```bash
jb clean doc/
```

If jupyter execution is cached, this command will not delete the cached folder. To remove the build folder (including `cached` executables), you can run:
```bash
jb clean --all doc/
```

## References
