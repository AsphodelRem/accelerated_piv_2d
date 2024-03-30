from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import os


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class BuildCMakeExt(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)
        super().run()

    def build_cmake(self, ext):
        import subprocess
        import os

        cwd = os.getcwd()
        build_temp = os.path.join(cwd, self.build_temp)
        os.makedirs(build_temp, exist_ok=True)

        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + os.path.abspath(self.build_lib),
            '-DPYTHON_EXECUTABLE=' + sys.executable,
            ]

        build_args = [
            '--config', 'Release',
        ]

        os.chdir(build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=build_temp)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=build_temp)
        os.chdir(cwd)

setup(
    name='accelerated_piv_cpp',
    version='0.1',
    author='Asphodel Rem',
    description=' ',
    ext_modules=[CMakeExtension('accelerated_piv_cpp')],
    cmdclass=dict(build_ext=BuildCMakeExt),
    zip_safe=False,
)
