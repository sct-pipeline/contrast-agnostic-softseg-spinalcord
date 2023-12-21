from setuptools import setup, find_packages

setup(
    name='contrast-agnostic-inference',
    version='0.1',
    author='https://github.com/sct-pipeline/contrast-agnostic-softseg-spinalcord/graphs/contributors',
    author_email='aac@example.com',
    packages=find_packages(),
    url='https://github.com/sct-pipeline/contrast-agnostic-softseg-spinalcord',
    license='MIT',
    description='Inference code for the contrast-agnostic spinal cord segmentation using SoftSeg',
    long_description=open('README.md').read(),
    install_requires=[],
    entry_points={
        'console_scripts': [
            'run_inference_single_image = run_inference_single_image:main',
            ]
        }
    )

