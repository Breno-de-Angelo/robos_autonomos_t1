import rclpy
import numpy as np
import cv2

from rclpy.node import Node, Publisher, Subscription
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class Pilot(Node):
    goal_pose_pub_: Publisher
    map_sub_: Subscription
    current_map: OccupancyGrid | None

    def __init__(self):
        super().__init__('pilot')

        self.declare_parameter('debug_cv', False)
        self.debug_cv = self.get_parameter('debug_cv').value

        self.goal_pose_pub_ = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.map_sub_ = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        if self.debug_cv:
            self.contour_pub_ = self.create_publisher(Image, '/contour_image', 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.current_map = None
        self.timer = self.create_timer(15.0, self.process_map)

        self.get_logger().info(f'Debug CV Mode: {"Enabled" if self.debug_cv else "Disabled"}')

    def map_callback(self, msg: OccupancyGrid):
        self.get_logger().info('Received map')
        self.current_map = msg

    def get_robot_position_map_image_frame(self, robot_tf_x,  robot_tf_y):
        x_robot = robot_tf_x - self.current_map.info.origin.position.x
        y_robot = robot_tf_y - self.current_map.info.origin.position.y
        x_map = int(x_robot / self.current_map.info.resolution)
        y_map = int(y_robot / self.current_map.info.resolution)
        return x_map, y_map

    def get_known_map_binary_image(self):
        """Returns a binary image where 0 is unknown and 255 is known"""
        width = self.current_map.info.width
        height = self.current_map.info.height
        data = np.array(self.current_map.data, dtype=np.int8).reshape((height, width))
        img = np.zeros((data.shape[0], data.shape[1]), dtype=np.uint8)
        img[data == -1] = 0     # Unknown cells -> Black
        img[data >= 0] = 255    # Known cells -> White
        return img

    def get_occupancy_map_matrix(self):
        """Returns an array 0 is free, 255 is occupied and 127 is unknown"""
        width = self.current_map.info.width
        height = self.current_map.info.height
        data = np.array(self.current_map.data, dtype=np.int8).reshape((height, width))
        data_scaled = np.where(data >= 0, (data / 100.0 * 255).astype(np.uint8), -1)
        return data_scaled

    def get_occupancy_value(self, occupancy_matrix, x, y, kernel_size=5):
        """Instead of simply getting the value of occupancy at (x, y), it gets the average of the 8 surrounding cells"""
        self.get_logger().info(f'Getting occupancy value at ({x}, {y})')

        if x == 0 or y == 0 or x == occupancy_matrix.shape[0] - 1 or y == occupancy_matrix.shape[1] - 1:
            self.get_logger().info('Out of bounds')
            return 255

        neighborhood = occupancy_matrix[x - kernel_size : x + kernel_size, y - kernel_size : y + kernel_size]
        valid_values = neighborhood[neighborhood >= 0]
        self.get_logger().info(f'Neighborhood: {neighborhood}')
        self.get_logger().info(f'Mean: {np.mean(valid_values) if valid_values.size > 0 else 255}')
        return np.mean(valid_values) if valid_values.size > 0 else 255

    def find_closest_point(self, robot_position, contours, occupancy_matrix, epsilon=10, occupancy_threshold=10):
        min_dist = float('inf')
        closest_point = None
        for contour in contours:
            for point in contour:
                px, py = point[0]
                dist = np.hypot(px - robot_position[0], py - robot_position[1])
                if epsilon < dist < min_dist:
                    min_dist = dist
                    if self.get_occupancy_value(occupancy_matrix, px, py) < occupancy_threshold:
                        closest_point = (px, py)
        return closest_point

    def get_goal_pose(self, closest_point):
        goal_pose = PoseStamped()
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.header.frame_id = "map"
        goal_pose.pose.position.x = closest_point[0] * self.current_map.info.resolution + self.current_map.info.origin.position.x
        goal_pose.pose.position.y = closest_point[1] * self.current_map.info.resolution + self.current_map.info.origin.position.y
        goal_pose.pose.orientation.w = 1.0  # No rotation
        return goal_pose

    def publish_contour_image(self, img, contours, robot_position, closest_point):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        cv2.drawContours(img_rgb, contours, -1, (0, 0, 255), 1)
        cv2.circle(img_rgb, robot_position, 1, (255, 0, 0), -1)
        cv2.circle(img_rgb, closest_point, 1, (0, 255, 0), -1)
        img_rgb = np.flipud(img_rgb)
        img_rgb = np.rot90(img_rgb)
        bridge = CvBridge()
        img_msg = bridge.cv2_to_imgmsg(img_rgb, encoding="bgr8")
        self.contour_pub_.publish(img_msg)

    def find_goal(self):
        binary_known_cells_map = self.get_known_map_binary_image()
        occupancy_matrix = self.get_occupancy_map_matrix()
        contours, _ = cv2.findContours(binary_known_cells_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        try:
            t = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0))
        except TransformException as ex:
            self.get_logger().info(f'Could not transform map to base_link: {ex}')
            return

        robot_position = self.get_robot_position_map_image_frame(t.transform.translation.x, t.transform.translation.y)
        closest_point = self.find_closest_point(robot_position, contours, occupancy_matrix)
        if closest_point is None:
            self.get_logger().info('No valid goal found')
            return

        if self.debug_cv:
            self.publish_contour_image(binary_known_cells_map, contours, robot_position, closest_point)

        goal_pose = self.get_goal_pose(closest_point)
        self.goal_pose_pub_.publish(goal_pose)
        self.get_logger().info(f'Published goal at: {goal_pose.pose.position.x}, {goal_pose.pose.position.y}')

    def process_map(self):
        if self.current_map is None:
            self.get_logger().info('Map not received yet')
            return
        self.get_logger().info('Processing map')
        self.find_goal()

def main(args=None):
    rclpy.init(args=args)
    node = Pilot()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
