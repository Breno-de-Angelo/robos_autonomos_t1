import rclpy
import numpy as np
import cv2
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from tf2_ros import TransformException, Buffer, TransformListener
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class HeuristicCostmap(Node):
    def __init__(self):
        super().__init__('heuristic_costmap')

        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.costmap_sub = self.create_subscription(OccupancyGrid, '/global_costmap/costmap', self.costmap_callback, 10)
        self.costmap_pub = self.create_publisher(OccupancyGrid, '/heuristic_costmap', 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.bridge = CvBridge()

        self.current_map = None
        self.current_costmap = None
        self.timer = self.create_timer(5.0, self.process_costmap)

    def map_callback(self, msg: OccupancyGrid):
        self.current_map = msg

    def costmap_callback(self, msg: OccupancyGrid):
        self.current_costmap = msg

    def get_cost_heuristic(self, costmap_matrix, x, y, dist):
        height, width = costmap_matrix.shape
        
        # Verifica se o índice está dentro dos limites da matriz
        if x < 0 or x >= width or y < 0 or y >= height:
            return 100  # Retorna um valor alto para posições inválidas
        
        if costmap_matrix[y, x] < 0:
            return 100  # Obstáculo ou desconhecido
        
        self.get_logger().info(f'costmap_matrix[y, x]: {costmap_matrix[y, x]}')
        dist = dist * self.current_map.info.resolution
        a = 100
        b = 0.6
        c = 1
        d = 2.0
        local_cost = int(100 * max(1 - dist / d, 0))
        distant_cost = int(a * np.tanh(max(b * (dist - d), 0)))
        cost = local_cost + distant_cost
        cost = local_cost + distant_cost + c * costmap_matrix[y, x] + 1
        # cost = costmap_matrix[y, x]

        if cost > 100:
            cost = 100
        return cost

    def process_costmap(self):
        if not self.current_map or not self.current_costmap:
            self.get_logger().info('Map(s) not received yet')
            return

        try:
            t = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0))
        except TransformException as ex:
            self.get_logger().warn(f'Transform failed: {ex}')
            return

        robot_x = t.transform.translation.x
        robot_y = t.transform.translation.y
        x_origin = self.current_map.info.origin.position.x
        y_origin = self.current_map.info.origin.position.y
        res = self.current_map.info.resolution

        robot_x_idx = int((robot_x - x_origin) / res)
        robot_y_idx = int((robot_y - y_origin) / res)

        width = self.current_map.info.width
        height = self.current_map.info.height
        costmap_matrix = np.array(self.current_costmap.data, dtype=np.int8).reshape((height, width))
        heuristic_map = np.zeros_like(costmap_matrix, dtype=np.int8)

        for x in range(width):
            for y in range(height):
                dist = np.hypot(x - robot_x_idx, y - robot_y_idx)
                heuristic_map[y, x] = self.get_cost_heuristic(costmap_matrix, x, y, dist)

        heuristic_costmap_msg = OccupancyGrid()
        heuristic_costmap_msg.header.stamp = self.get_clock().now().to_msg()
        heuristic_costmap_msg.header.frame_id = 'map'
        heuristic_costmap_msg.info = self.current_map.info
        heuristic_costmap_msg.data = heuristic_map.flatten().tolist()

        self.costmap_pub.publish(heuristic_costmap_msg)
        self.get_logger().info('Published heuristic costmap')


def main(args=None):
    rclpy.init(args=args)
    node = HeuristicCostmap()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
