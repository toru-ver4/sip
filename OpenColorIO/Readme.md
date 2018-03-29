# OCIOを使った独自コンフィグの作成

## 概要
NUKEでHDRのOutputを行うための ```ocio.config``` を作成する。

## ゴール

優先順位の高い順に書いたのが以下。

* ST2084, BT.2020, D65 の Output の定義作成
* SDR を 0.1 にマッピングして import する定義作成

## ソースコード類
[ACES 1.0.3 OpenColorIO configuration](https://github.com/imageworks/OpenColorIO-Configs/tree/master/aces_1.0.3)


## Install on Ubuntu16.04 on Windows10

### Install OpenImageIO

```bash
$ sudo yum -y install OpenImageIO-devel
$ sudo yum -y install python-OpenImageIO
```

### Install OpenColorIO
```bash
$ sudo yum -y install OpenColorIO-devel OpenColorIO-tools
```

### Install Optional Dependencies
```bash
$ sudo yum -y install LibRaw-devel opencv-devel OpenEXR-devel 
```

### Install CTL

```bash
$ sudo yum -y install ilmbase-devel libtiff 
$ mkdir -p ~/local/src
$ cd ~/local/src
$ wget ftp://download.osgeo.org/libtiff/tiff-4.0.9.tar.gz
$ tar xvzf tiff-4.0.9.tar.gz
$ cd tiff-4.0.9
$ ./configure
$ make
$ sudo make install
$ cd ~/local/src
$ git clone https://github.com/ampas/aces_container.git
$ cd aces_container
$ mkdir build && cd build
$ cmake ..
$ make
$ sudo make install
$ cd ~/local/src
$ git clone https://github.com/ampas/CTL.git
$ cd CTL
$ mkdir build && cd build
$ cmake ..
$ make
$ sudo make install
```

## Download the ctl files
https://github.com/ampas/aces-dev/tree/master/transforms/ctl

## ocio.config 作成

```bash
./create_aces_config -a ../../../transforms/ctl -c ../../../ocio_config --lutResolution1d 4096 --lutResolution3d 65 --keepTempImages
```