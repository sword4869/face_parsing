from setuptools import setup

setup(
    name='face_parsing',
    version='0.2',
    description='This repo is used to generate semantic segmentation for face image',
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author='sword4869',
    url='https://github.com/sword4869/face_parsing',
    install_requires=[
        'configargparse',
        'opencv-contrib-python',
        'numpy',
        'pillow',
        'tqdm',
        'torch',
        'torchvision'
    ],
    entry_points={
        'console_scripts': [
            'face_parsing = face_parsing.segment:run_cli',
        ]
    },
)