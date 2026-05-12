from setuptools import find_packages, setup

package_name = 'robot_motion'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (
            'share/' + package_name + '/launch',
            [
                'launch/person_follow.launch.py',
                'launch/person_follow_lidar.launch.py',
                'launch/person_follow_debug.rviz',
            ],
        ),
        ('share/' + package_name + '/config', ['config/arm_poses.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robohack2026',
    maintainer_email='dev@robohack2026.local',
    description='Person following and motion control nodes for Agibot X2',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'head_tracker   = robot_motion.head_tracker_node:main',
            'body_follower  = robot_motion.body_follower_node:main',
            'arm_pose       = robot_motion.arm_pose_node:main',
            'person_follow  = robot_motion.person_follow_node:main',
        ],
    },
)
