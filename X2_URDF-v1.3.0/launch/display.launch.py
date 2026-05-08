import os
from struct import pack

from setuptools import Command

from ament_index_python import get_package_share_directory, get_package_share_path

import launch
import launch.launch_description_sources
import launch.substitutions
import launch_ros
import launch_ros.substitutions

def generate_launch_description():

    turtlebot2_description_package = launch_ros.substitutions.FindPackageShare(package='x2_description').find('x2_description')
    # ref:https://answers.ros.org/question/417369/caught-exception-in-launch-see-debug-for-traceback-unable-to-parse-the-value-of-parameter-robot_description-as-yaml/
    robot_state_publisher_node = launch_ros.actions.Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': launch_ros.descriptions.ParameterValue( launch.substitutions.Command(['cat ',os.path.join(turtlebot2_description_package,'urdf/x2t1.urdf'),]), value_type=str)  }]
    )

    joint_state_publisher_node = launch_ros.actions.Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui'
    )
    rviz_node = launch_ros.actions.Node(
        package="rviz2",
        executable="rviz2",
        name="rivz2",
        output="screen",
        arguments=["-d",os.path.join(turtlebot2_description_package,'launch/config.rviz')]
    )




    return launch.LaunchDescription([
        robot_state_publisher_node,
        joint_state_publisher_node,
        rviz_node,
    ])