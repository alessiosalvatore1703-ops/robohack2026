from setuptools import find_packages, setup

package_name = 'robot_vision'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/stereo_detector.launch.py']),
        ('share/' + package_name + '/config', ['config/stereo_detector.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robohack2026',
    maintainer_email='dev@robohack2026.local',
    description='Stereo YOLO person detection pipeline for Agibot X2',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'stereo_detector  = robot_vision.stereo_detector_node:main',
            'view_stereo      = robot_vision.view_stereo_node:main',
        ],
    },
)
