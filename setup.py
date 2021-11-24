## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD

from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

# fetch values from package.xml
setup_args = generate_distutils_setup(
        name='dnn_experiment',
        packages=['bonet'],  # list of package name
        package_dir={'bonet':'./bonet'
        } # path of each package
)
setup(**setup_args)

