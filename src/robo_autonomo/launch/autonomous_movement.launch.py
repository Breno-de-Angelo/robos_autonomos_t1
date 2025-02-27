import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch.actions import TimerAction


def generate_launch_description():
    package_name = 'robo_autonomo'
    os.environ['TURTLEBOT3_MODEL'] = 'waffle'
    try:
        os.environ['GAZEBO_MODEL_PATH'] = os.environ['GAZEBO_MODEL_PATH'] + ':' + os.path.join(get_package_share_directory('turtlebot3_gazebo'), 'models')
    except KeyError:
        os.environ['GAZEBO_MODEL_PATH'] = os.path.join(get_package_share_directory('turtlebot3_gazebo'), 'models')

    this_package_dir = get_package_share_directory(package_name)
    turtlebot3_world_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([this_package_dir, '/launch/turtlebot3_world.launch.py'])
    )
    
    slam_toolbox_launch_dir = get_package_share_directory('slam_toolbox')
    slam_toolbox_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([slam_toolbox_launch_dir, '/launch/online_async_launch.py']),
        launch_arguments={'use_sim_time': 'true'}.items()
    )
    
    nav2_launch_dir = get_package_share_directory('nav2_bringup')
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([nav2_launch_dir, '/launch/navigation_launch.py']),
        launch_arguments={'use_sim_time': 'true', 'params_file': f"{this_package_dir}/config/nav2_params.yaml"}.items()
    )

    pointcloud_to_laserscan_node = Node(
        package='pointcloud_to_laserscan',
        executable='pointcloud_to_laserscan_node',
        name='pointcloud_to_laserscan',
        remappings=[
            ('cloud_in', '/velodyne2/velodyne_points2'),
            ('scan', '/scan'),
        ],
        parameters=[{
            'target_frame': 'base_scan',
            # 'max_height': 0.1
        }],
        output='screen',
    )

    pilot_node = Node(
        package=package_name,
        executable="pilot",
        name="pilot",
        parameters=[{"use_sim_time": True, "debug_cv": True}],
    )

    item_detector_node = Node(
        package=package_name,
        executable="item_detector_camera_and_lidar",
        name="item_detector_camera_and_lidar",
        parameters=[{"use_sim_time": True, "debug": True}],
    )

    rviz_config_file = os.path.join(
        this_package_dir,
        'config',
        'teste.rviz'
    )
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file]
    )

    return LaunchDescription([
        turtlebot3_world_launch,
        pointcloud_to_laserscan_node,
        slam_toolbox_launch,
        nav2_launch,
        TimerAction(period=10.0,
                    actions=[pilot_node]),
        TimerAction(period=5.0,
                    actions=[item_detector_node]),
        rviz_node,
    ])
