# Robôs Autônomos - Trabalho 1

O projeto funciona com o ros2 foxy e humble. A partir do ros2 jazzy, o gazebo alterou do gazebo classic para o gazebo harmonic o que necessita de adaptações para rodar.

O tutorial a seguir mostra como rodar usando Docker. Caso tenha uma das versões do ros já instalada na máquina, você pode pular a preparação do ambiente Docker.

Para iniciar, clone o repositório e entre na pasta do projeto:
```bash
git clone https://github.com/Breno-de-Angelo/robos_autonomos_t1
cd robos_autonomos_t1
```

# Preparação do ambiente Docker
```bash
docker pull osrf/ros:humble-desktop-full
xhost + local:docker
export DISPLAY=:1

docker run --name ros -it --net=host --device /dev/dri/ \
-e DISPLAY=$DISPLAY -v $HOME/.Xauthority:/root/.Xauthority:ro \
-v .:/root/robos_autonomos_t1 \
osrf/ros:humble-desktop-full
```

Dessa forma, você terá um container rodando o ros2 humble com a interface gráfica preparada. Para maior comodidade, instale a extensão do vscode para o docker e abra o container para facilitar a edição dos arquivos.

Como o diretório do repositório foi mapeado para o container, você pode editar os arquivos diretamente no vscode dentro do container e isso irá se refletir para fora.

Por fim, adicione no bashrc do container o source do ros para facilitar a vida.
```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

# Rodando o projeto
Primeiramente é necessário instalar as dependências do projeto:
```bash
apt update
rosdep update
rosdep install --from-paths src --ignore-src -r -y
```

Em seguida, compile o projeto:
```bash
colcon build --symlink-install
```

Para rodar a simulação, execute em terminais diferentes:
```bash
source install/setup.bash
export TURTLEBOT3_MODEL=waffle
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/opt/ros/humble/share/turtlebot3_gazebo/models
ros2 launch robo_autonomo autonomous_movement.launch.py
```

```bash
```
