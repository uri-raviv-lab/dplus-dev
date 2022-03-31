import os
import numpy
import shutil
import sys
from setuptools import setup
from distutils.extension import Extension
import setuptools
from setuptools.command.build_ext import build_ext as original_build_ext

with open(os.path.join(os.path.dirname(__file__), 'README.md')) as readme:
    README = readme.read()

with open(os.path.join(os.path.dirname(__file__), 'LICENSE.txt')) as license:
    LICENSE = license.read()

# On Windows, DEBUG means using the ReleaseWithDebugInfo DLLs, which allow debugging of the
# backend inside Visual Studio. On Linux it has no effect.
#
# There is no reasonable way to pass arguments to bdist_wheel, so on Windows, DEBUG is set to
# False unless the environment variable DPLUS_API_DEBUG is set to DEBUG.

if sys.platform == 'win32':
    debug_env = os.environ.get('DPLUS_API_DEBUG', "")
    DEBUG = debug_env.upper() =="DEBUG"
    VERSION_STR = debug_env.lower() #modify wheel name for dplus resources
    print('dplus-api Debug mode: ', DEBUG)
else:
    DEBUG = False
    VERSION_STR = ""

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # This is the project's root dir
API_DIR = os.path.dirname(__file__)
INCLUDE_DIRS = [ROOT_DIR, os.path.join(ROOT_DIR, 'Common')]
LIBRARY_DIRS = [os.path.join(ROOT_DIR, "x64", "ReleaseWithDebugInfo" if DEBUG else "Release")]
REQUIRED_DLLS = ['cudart64_110', 'curand64_10', 'lua51-backend', 'PDBReaderLib', 'xplusbackend']

extra_compile_args = []
extra_link_args = []
if sys.platform == 'win32':
    extra_compile_args = ['/Ox'] if not DEBUG else []
    LIBRARIES_DIR = os.path.join(ROOT_DIR, "x64", "ReleaseWithDebugInfo" if DEBUG else "Release")
    REQUIRED_DLLS = ['cudart64_110', 'curand64_10', 'lua51-backend', 'PDBReaderLib', 'xplusbackend']
    LIBRARIES = ['xplusbackend']
    # extra_link_args = ['/debug']
elif sys.platform == 'linux':
    extra_compile_args = ['-fPIC', '-std=c++14']
    LIBRARIES_DIR = os.path.join(API_DIR, 'lib')
    LIBRARIES = ['backend']

# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

class PrepareCommand(setuptools.Command):
    description = "Convert the pyx files to cpp so there's no cython dependence in installation"
    # user_options = [('debug', None, 'debug')]
    user_options = []

    def initialize_options(self):
        # self.debug = None
        pass

    def finalize_options(self):
        pass

    def run(self):
        print("running prepare command")
        first_pyx = os.path.join('dplus', 'wrappers', 'wrappers.pyx')
        self.convert_to_c(first_pyx)

        if sys.platform == 'win32':
            self.move_dlls()
        elif sys.platform in ['darwin', 'linux']:
            self.move_sos()


    def convert_to_c(self, pyx):
        #creates fast.h and fast.c in cpp_wrapper folder
        print('Converting dplus pyx files to C++ sources...')
        print(pyx)
        self.cython(pyx)
        print('Converting {} pyx files to C++ sources...'.format(pyx.split("/")[-1]))


    def cython(self, pyx):
        from Cython.Compiler.CmdLine import parse_command_line
        from Cython.Compiler.Main import compile
        options, sources = parse_command_line(['-2', '-v', '--cplus', pyx])
        result = compile(sources, options)
        if result.num_errors > 0:
            print('Errors converting %s to C++' % pyx, file=sys.stderr)
            raise Exception('Errors converting %s to C++' % pyx)
        self.announce('Converted %s to C++' % pyx)

    def move_dlls(self):
        if sys.platform != 'win32':
            raise NotImplemented('move_dlls is Windows specific')
        # Move DLLs so they can be included in the package.
        print('Copying necessary DLLs')
        for dll_filename in REQUIRED_DLLS:
            dll_filename = dll_filename + '.dll'
            shutil.copy(os.path.join(LIBRARY_DIRS[0], dll_filename), 'dplus')

    def move_sos(self):
        if sys.platform not in ['linux', 'darwin']:
            raise NotImplemented('move_sos only works on Linux and Macs')
        for filename in os.listdir(LIBRARIES_DIR):
            full_filename = os.path.join(LIBRARIES_DIR, filename)
            if os.path.isfile(full_filename):
                shutil.copy(full_filename, 'dplus')



setup(
    name='dplus-api',
    version=VERSION_STR if VERSION_STR else '4.7.1',
    packages=['dplus'],
    package_data= { 'dplus': ['*.dll'] if sys.platform == 'win32' else ['lib*.so*'] },
	install_requires=['numpy>=1.10', 'psutil>=5.6.3', 'requests>=2.10.0', 'pyceres>=0.1.0'],
    # include_package_data=True, # If True - ignores the package_data property.
    license=LICENSE,  # example license
    description='Call the DPlus Calculation Backend',
    url='https://scholars.huji.ac.il/uriraviv',
    author='Devora Witty',
    author_email='devorawitty@chelem.co.il',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: Other/Proprietary License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Chemistry',
    ],
    cmdclass={
        'prepare': PrepareCommand,
    },
    ext_modules=[
        Extension(
            "dplus.wrappers",
            ["dplus/wrappers/wrappers.cpp"],
            language='c++',
            include_dirs=INCLUDE_DIRS + [numpy.get_include()],
            library_dirs=[LIBRARIES_DIR],
            libraries=LIBRARIES,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args),
    ]
)
