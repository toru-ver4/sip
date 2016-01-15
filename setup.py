#-*- coding: utf-8 -*-
 
from distutils.core import setup
import py2exe

# option = {
#     "compressed"   : 1,
#     "optimize"     : 2,
#     "bundle_files" : 2
# }


# setup(
#     options = {"py2exe" : option },
#     console = ["matplot_test.py"],
#     zipfile = "matplot_test"
# )

setup(
    options = { "py2exe" : {"includes" : ["matplotlib.backends.backend_tkagg", "tkinter", "tkinter.filedialog"] } },
    console=["matplot_test.py"]

)
