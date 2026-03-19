import os
import sys
import subprocess
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import tomli

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())

class CMakeBuild(build_ext):
    def build_extension(self, ext):
        ext_dir = Path(self.get_ext_fullpath(ext.name)).parent.absolute()
        build_dir = Path(self.build_temp) / ext.name
        build_dir.mkdir(parents=True, exist_ok=True)

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={ext_dir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={'Debug' if self.debug else 'Release'}",
        ]

        # TODO: make platform-specific
        build_args = ["--", "-j2"]

        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=build_dir
        )

        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=build_dir
        )

        # The file should now be in ext_dir - no manual copying needed
        # expected = list(ext_dir.glob(f"{ext.name}*.so")) + list(ext_dir.glob(f"{ext.name}*.pyd"))
        # if not expected:
        #     raise RuntimeError(f"No module found in {ext_dir}")

# Read metadata
with open("pyproject.toml", "rb") as f:
    pyproject = tomli.load(f)
    project_metadata = pyproject.get("project", {})

setup(
    name=project_metadata["name"],
    version=project_metadata["version"],
    author=project_metadata["authors"][0]["name"],
    description=project_metadata["description"],
    ext_modules=[CMakeExtension('tempo')],
    cmdclass={'build_ext': CMakeBuild},
    zip_safe=False,
    python_requires='>=3.6',
)
