# OCIOを使った独自コンフィグの作成

## 概要
NUKEでHDRのOutputを行うための ```ocio.config``` を作成する。

## ゴール

優先順位の高い順に書いたのが以下。

* ST2084, BT.2020, D65 の Output の定義作成
* SDR を 0.1 にマッピングして import する定義作成

## Install on Ubuntu16.04 on Windows10

### Install CTL

1. Build libraries from source code

```bash
$ sudo apt-get install cmake libilmbase-dev libopenexr-dev libtiff5

# install aces container
$ mkdir -p ~/local/src
$ cd ~/local/src
$ git clone https://github.com/ampas/aces_container.git
$ cd aces_container
$ mkdir build && cd build
$ cmake ..
$ make
$ sudo make install

# install CTL
$ cd ~/local/src
$ git clone https://github.com/ampas/CTL.git
$ cd CTL
$ mkdir build && cd build
$ cmake ..
$ make
$ sudo make install
```

2. Install pre-build libraries

```bash
$ sudo apt-get install python-pyopencolorio python-openimageio
$ sudo apt-get install python-opencv
$ sudo apt-get install libraw-dev
```