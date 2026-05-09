from setuptools import find_packages, setup

package_name = 'yolo_person_detector'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/yolo_pipeline.launch.py']),
        ('share/' + package_name + '/config', ['config/yolo_params.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robohack2026',
    maintainer_email='dev@robohack2026.local',
    description='YOLO-based person detection pipeline for Agibot X2',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo_detector_node = yolo_person_detector.yolo_detector_node:main',
            'camera_selector_node = yolo_person_detector.camera_selector_node:main',
            'visualization_node = yolo_person_detector.visualization_node:main',
        ],
    },
)
