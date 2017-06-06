# INSTALL

## OpenImageIO

see the INSTALL.md of the [OpenImageIO official site](https://github.com/OpenImageIO/oiio).

### Basic packages

```
$ sudo apt-get install libboost-dev libtiff5-dev libopenexr-dev libjpeg-dev libpng16-dev
$ sudo apt-get install qt5-default glew-utils libboost-all-dev 0libqt4-dev
$ sudo apt-get install python3-all-dev
```


### libboost for python3
```
$ git clone --recursive -b boost-1.60.0 https://github.com/boostorg/boost.git
$ cd boost
$ git checkout develop
$ ./bootstrap.sh --with-libraries=python --with-python=python3 --with-python-version=3.5
$ ./b2
$ ./b2 headers
$ sudo ./b2 install

```

```
$ source ~/.bashrc
```

### OpenImageIO

modify the src/python/CMakeLists.txt

```
diff --git a/src/python/CMakeLists.txt b/src/python/CMakeLists.txt
index 13e62d0..09523fd 100644
--- a/src/python/CMakeLists.txt
+++ b/src/python/CMakeLists.txt
@@ -42,7 +42,7 @@ else ()
     #Finding the python3 component for boost is a little tricky, since it has
     #different names on different systems. Try the most common ones
     #(boost_python3, boost_python-py34, …).
-    foreach (_boost_py3_lib python3 python-py34 python-py33 python-py32)
+    foreach (_boost_py3_lib python3 python-py35 python-py34 python-py33 python-py32)
         find_package (Boost 1.42 QUIET COMPONENTS ${_boost_py3_lib})
         string (TOUPPER ${_boost_py3_lib} boost_py3_lib_name)
         if (Boost_${boost_py3_lib_name}_FOUND)
```

```
$ git clone https://github.com/OpenImageIO/oiio.git
$ cd oiio
$ make -j8 VERBOSE=1 STOP_ON_WARNING=0 USE_PYTHON3=1 INSTALL_PREFIX=$HOME/local/python3 
```

### update PYTHONPATH

```
$ export PYTHONPATH=${PYTHONPATH}:$HOME/local/python3/python3
```