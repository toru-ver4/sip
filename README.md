# sip
sample code for SIgnal Processing.

## requirements
 - Anaconda(Python 3.5>)

## install 
Get the Anaconda installer from https://www.continuum.io/downloads .
Exec the installer. Please select the default settings in the installer.

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

# Appendix

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
2. open the article.tplx, s/{article}/{ltjsarticle}
3. save this file on ** Another Name ** (ex. ltjsarticle.tplx).

### convert

run the following commmand.

```bat
> jupyter nbconvert --to latex --template ltjsarticle.tplx lualatex [jupyter_file].ipynb
> lualatex sample_codes.tex
```


