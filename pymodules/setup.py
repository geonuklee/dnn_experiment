from distutils.core import setup

setup(name='',
      version='1.0',
      description='Python Distribution Utilities',
      author='Geonuk Lee',
      packages=['unet', 'Objectron', 'bonet_dataset'],
      package_dir={
        'unet' : './unet',
        'Objectron' : './Objectron',
        'bonet_dataset':'./bonet_dataset',
        },
     )
