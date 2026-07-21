"""
skeleton_pipeline.launch.py

Sobe dois processos de uma vez:
  1. skeleton_publisher  → o nó que processa o dataset e publica /skeletons_3d
  2. rviz2                → já aberto com um arquivo de config apontando
                             pro tópico certo (evita ter que configurar
                             manualmente toda vez)

Uso:
    ros2 launch skeleton_ros2 skeleton_pipeline.launch.py dataset_name:=shelf
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    dataset_name_arg = DeclareLaunchArgument(
        'dataset_name', default_value='shelf',
        description='Nome do dataset definido em etc/config.json'
    )
    publish_rate_arg = DeclareLaunchArgument(
        'publish_rate_hz', default_value='10.0',
        description='Frequência de publicação do MarkerArray'
    )

    skeleton_publisher_node = Node(
        package='skeleton_ros2',
        executable='skeleton_publisher',
        name='skeleton_publisher',
        output='screen',   # imprime os logs do nó no terminal
        parameters=[{
            'dataset_name': LaunchConfiguration('dataset_name'),
            'publish_rate_hz': LaunchConfiguration('publish_rate_hz'),
            'frame_id': 'world',
        }],
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        # Sem -d, o RViz2 abre em branco e você configura manualmente
        # (Add > By topic > /skeletons_3d > MarkerArray, Fixed Frame = world).
        # Depois de configurar do jeito que gosta, salve com
        # File > Save Config As... em src/skeleton_ros2/rviz/skeleton_view.rviz
        # e descomente a linha abaixo para carregar automaticamente:
        # arguments=['-d', os.path.join(
        #     get_package_share_directory('skeleton_ros2'), 'rviz', 'skeleton_view.rviz'
        # )],
    )

    return LaunchDescription([
        dataset_name_arg,
        publish_rate_arg,
        skeleton_publisher_node,
        rviz_node,
    ])
