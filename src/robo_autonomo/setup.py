from setuptools import setup
import os
from glob import glob

package_name = "robo_autonomo"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/" + package_name, ["package.xml"]),  # Include package.xml
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),  # Include launch files
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Your Name",
    maintainer_email="your.email@example.com",
    description="Robo Autonomo package",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            # Add your executable scripts here if needed
            'send_goal = robo_autonomo.send_goal:main',
        ],
    },
)
