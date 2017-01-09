# sip
sample code for SIgnal Processing.

# Installation Guide for Winodws
## requirements
 - Anaconda(Python 3.5>)

## install 
Get the Anaconda installer from https://www.continuum.io/downloads .
Exec the installer. Please select the default settings in the installer.

## add PYTHONPATH
set `./lib` to the PYTHONPATH

## make virtual environment.

```
> conda create -n default python=3.5 anaconda
```

## change the virtual environment.
```
> activate default # if in the Linux Environment, please type "source activate default"
```

## setup jupyter 

```
> jupyter notebook --generate-congif
```

change the modify the 'notebook_dir' parameters from '' to 'C:\home'.


## run jupyter
```
> jupyter notebook
```

# Installation Guide for Linux

## requirements
 - [pyenv](https://github.com/yyuu/pyenv)
 - [pyenv-virtualenv](https://github.com/yyuu/pyenv-virtualenv)

## install pyenv
see Readme.md of the [Github](https://github.com/yyuu/pyenv).

## install pyenv-virtualenv
see Readme.md of the [Github](https://github.com/yyuu/pyenv-virtualenv).

## install libraries

```bash:install.sh
$ sudo apt-get update
$ sudo apt-get install libssl-dev libbz2-dev libreadline-dev libsqlite3-dev -y 
```
## setup the virtual environment

```bash:install.sh
$ pyenv install -l
$ pyenv install anaconda3-x.x.x
$ pyenv virtualenv anaconda3-x.x.x my-virtual-env-3-x.x.x
```

## switch the environment

```bash:change.sh
$ pyenv virtualenvs  # check virtualenv
$ pyenv local anaconda3-x.x.x/envs/my-virtual-env-3-x.x.x
```

## add PYTHONPATH
```
export PYTHONPATH="./lib:$PYTHONPATH"
```

# Appendix

## Install OpenCV

```
> conda install -c https://conda.anaconda.org/menpo opencv3 
```

## convert from .ipynb to .html

run the following commmand.

```bat
> jupyter nbconvert --to html sample_codes.ipynb
```

## convert from .ipynb to .pdf

### install LaTeX package.

install LuaTeX. see [THIS PAGE](https://texwiki.texjp.org/?TeX%20Live).

### modify the latex configuration files
1. search the 'article.tplx' in the Anaconda3 directory.
2. open the article.tplx, s/{article}/{ltjsarticle}/
3. save this file on ** Another Name ** (ex. ltjsarticle.tplx).

### convert (command line)

run the following commmand.

```bat
> jupyter nbconvert --to latex --template ltjsarticle.tplx lualatex [jupyter_file].ipynb
> lualatex sample_codes.tex
```

### convert (Jupyter Notebook)

1. change 'article.tplx', s/{article}/{ltjsarticle}/
2. change 'pdf.py', s/u"pdflatex"/u"lualatex"/


