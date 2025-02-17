from setuptools import setup
from glob import glob
import os

package_name = "robo_autonomo"

# Função para mapear cada arquivo para seu respectivo diretório dentro do `share/` no ROS
def get_model_data_files():
    data_files = []
    base_dir = "models/"
    
    for root, _, files in os.walk(base_dir):
        if files:
            dest = os.path.join("share", package_name, root)  # Mantém a estrutura de diretórios
            src_files = [os.path.join(root, f) for f in files]  # Lista de arquivos completos
            data_files.append((dest, src_files))

    return data_files

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml") + glob("config/*.rviz")),
        ("share/" + package_name + "/worlds", glob("worlds/*.world") + glob("worlds/group/*")),
    ] + get_model_data_files(),  # Adiciona arquivos mantendo estrutura
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Breno de Angelo",
    maintainer_email="brenodeangelo@gmail.com",
    description="Robo Autonomo package",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            # Add your executable scripts here if needed
            'pilot = robo_autonomo.pilot:main',
            'heuristic_tuning = robo_autonomo.heuristic_tuning:main',
            'item_detector_camera_and_lidar = robo_autonomo.item_detector_camera_and_lidar:main',
        ],
    },
)
