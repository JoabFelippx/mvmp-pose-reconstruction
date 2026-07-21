import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'skeleton_ros2'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # Registra o pacote no índice do ROS 2 (obrigatório)
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        (os.path.join('share', package_name), ['package.xml']),
        # Instala todos os arquivos de launch/*.py junto com o pacote,
        # assim `ros2 launch skeleton_ros2 skeleton_pipeline.launch.py` funciona.
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*.launch.py'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Pessoa',
    maintainer_email='pessoa@ifes.edu.br',
    description='Publica esqueletos 3D reconstruídos como MarkerArray para RViz2.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # formato: 'nome_do_comando = pacote.modulo:funcao_main'
            'skeleton_publisher = skeleton_ros2.skeleton_publisher_node:main',
        ],
    },
)
