from setuptools import setup
import os
from glob import glob

package_name = "robo_autonomo"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml") + glob("config/*.rviz")),
    ],
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
            'send_goal = robo_autonomo.send_goal:main',
            'pilot = robo_autonomo.pilot:main',
            'heuristic_tuning = robo_autonomo.heuristic_tuning:main',
        ],
    },
)
