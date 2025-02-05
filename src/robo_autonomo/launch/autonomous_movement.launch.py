from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    turtlebot3_gazebo_launch_dir = FindPackageShare(package='turtlebot3_gazebo').find('turtlebot3_gazebo')
    turtlebot3_world_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([turtlebot3_gazebo_launch_dir, '/launch/turtlebot3_world.launch.py'])
    )
    slam_toolbox_launch_dir = FindPackageShare(package='slam_toolbox').find('slam_toolbox')
    slam_toolbox_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([slam_toolbox_launch_dir, '/launch/online_async_launch.py']),
        launch_arguments={'use_sim_time': 'true'}.items()
    )
    nav2_launch_dir = FindPackageShare(package='nav2_bringup').find('nav2_bringup')
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([nav2_launch_dir, '/launch/navigation_launch.py']),
        launch_arguments={'use_sim_time': 'true'}.items()
    )
    return LaunchDescription([
        turtlebot3_world_launch,
        slam_toolbox_launch,
        nav2_launch,
    ])