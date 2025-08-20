from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='bird_camera',
            executable='zoom_control',
            name='zoom_control_node',
            output='screen',
            parameters=[{
                "serial_port": "/dev/ttyUSB0",
                "initial_zoom": 0,
                "reset_to_1x_on_start": True
            }]

        )
    ])
