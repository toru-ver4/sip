#!/bin/bash
python make_ocio_luts.py
export PYTHONPATH=/usr/local/lib/python2.7/site-packages:/work/src/lib
python2 make_ocio_config.py
