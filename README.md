# Notes on Stats
This repository collects some statistical tools we might use during projects of computed vision based behavioral phenotyping. 

Durham 2020 
matias.di.martino.uy@gmail.com  


## Index:
For the moment, just follow the python nootbook notebooks/dev.ipynb, it is pretty self-explanatory. 

## License information.

## Installation Instructions.
0. Prerequisited:
- Python 3, pip, and virtualenvwrapper (if you don't have them installed look at the next section for instructions). 
1. Prepare your python environment.  
$mkvirtualenv ENVNAME -p python3 &nbsp; &nbsp; &nbsp; _create your virtual environment_.  
$workon ENVNAME &nbsp; &nbsp; &nbsp; _use your brand new virtual environment_.  
$pip install -r requirements.txt &nbsp; &nbsp; &nbsp; _install python libraries_.  
2. Get our code.  
$git clone XXXXXX  

## References.



## Installing pre-requisites: (this is for mac, if you are in linux is similar, in windows idk)
Install python:  
$brew install python.  
Check that it is correctly installed. 
$python3 --version.     
$Python 3.x.x     

Install pip:  
$sudo easy_install pip    

Install virtualenvwrapper:
$sudo pip install virtualenvwrapper   (maybe you have to use pip3 to install it in python3)

Finally add Shell Startup File
Add three lines to your shell startup file (.zshrc, .bashrc, .profile, etc.) to set the location where the virtual environments should live, the location of your development project directories, and the location of the script installed with this package:

export WORKON_HOME=$HOME/.virtualenvs
export PROJECT_HOME=$HOME/Devel
source /usr/local/bin/virtualenvwrapper.sh

