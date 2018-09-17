Prerequisites:
--------------
For compiling on Linux:
> pip install cython

--------------
For compiling on windows (64 bit), refer to:
https://github.com/cython/cython/wiki/CythonExtensionsOnWindows

> set DISTUTILS_USE_SDK=1
> setenv /x64 /release

==============
To build cpp arms:
run:
> python setup.py build_ext -i
from proper folder

==============
For numpy include errors:
Add 'include_dirs=[numpy.get_include()]' to ext_modules in setup.py
(and 'include numpy' at the top)
