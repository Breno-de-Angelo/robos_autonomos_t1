import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from sklearn.cluster import KMeans

class CubeDetector(Node):
    def __init__(self):
        super().__init__('cube_detector')
        self.declare_parameter('debug', False)

        self.debug = self.get_parameter('debug').value
        self.bridge = CvBridge()

        # Inscreve-se no tópico da câmera
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',  # Substitua pelo seu tópico de câmera
            self.image_callback,
            10)
        self.subscription  # Evita warning de variável não usada

        # Se debug=True, publica a imagem processada
        if self.debug:
            self.debug_publisher = self.create_publisher(Image, '/debug_detection', 10)

    def image_callback(self, msg):
        # Converte a imagem ROS2 para OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # Reduz ruído
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)

        # Converte para HSV
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Amostra pixels para encontrar a cor predominante usando KMeans
        pixels = hsv.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, n_init=10)
        kmeans.fit(pixels)
        dominant_color = np.uint8(kmeans.cluster_centers_[0])

        # Cria uma máscara com base na cor predominante
        lower_bound = np.array([max(0, dominant_color[0] - 10), 50, 50])
        upper_bound = np.array([min(180, dominant_color[0] + 10), 255, 255])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # Encontra contornos da máscara
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Seleciona o maior contorno
            max_contour = max(contours, key=cv2.contourArea)

            # Aproxima o contorno
            approx = cv2.approxPolyDP(max_contour, 0.02 * cv2.arcLength(max_contour, True), True)

            # Se tiver 4 lados, pode ser o cubo
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                self.get_logger().info(f"Cubo detectado: x={x}, y={y}, w={w}, h={h}")

        # Publica a imagem processada se debug=True
        if self.debug:
            debug_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            self.debug_publisher.publish(debug_msg)

def main(args=None):
    rclpy.init(args=args)
    detector = CubeDetector()
    rclpy.spin(detector)
    detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
