import rclpy
import rclpy.logging
import rclpy.time
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
    current_costmap: OccupancyGrid | None
    no_valid_goal_count: int

    def __init__(self):
        super().__init__('pilot')

        self.declare_parameter('debug_cv', False)
        self.debug_cv = self.get_parameter('debug_cv').value

        self.goal_pose_pub_ = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.map_sub_ = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.costmap_sub_ = self.create_subscription(OccupancyGrid, '/global_costmap/costmap', self.costmap_callback, 10)
        if self.debug_cv:
            self.contour_pub_ = self.create_publisher(Image, '/contour_image', 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.current_map = None
        self.current_costmap = None
        self.no_valid_goal_count = 0
        self.timer = self.create_timer(15.0, self.process_map)

        self.get_logger().info(f'Debug CV Mode: {"Enabled" if self.debug_cv else "Disabled"}')

    def map_callback(self, msg: OccupancyGrid):
        self.get_logger().info('Received map')
        self.current_map = msg

    def costmap_callback(self, msg: OccupancyGrid):
        self.get_logger().info('Received costmap')
        self.current_costmap = msg

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

    def get_costmap_matrix(self):
        """Returns an array 0 is free, 255 is occupied and 127 is unknown"""
        width = self.current_costmap.info.width
        height = self.current_costmap.info.height
        data = np.array(self.current_costmap.data, dtype=np.int8).reshape((height, width))
        return data

    def get_cost_heuristic(self, costmap_matrix, x, y, robot_position, dist):
        """
        Heuristic function to get the cost of a point based on its distance to the robot and global costmap.
        The farther the point is to the robot and the higher the occupancy value, the higher the cost.
        """
        if costmap_matrix[y, x] < 0:
            return 100
        # a = 20
        # b = 0.4
        # c = 1
        # d = 2.0
        # local_cost = int(100 * max(1 - dist / d, 0))
        # distant_cost = int(a * np.tanh(max(b * (dist - d), 0)))
        # cost = local_cost + distant_cost + c * costmap_matrix[y, x]
        cost = costmap_matrix[y, x]
        if cost > 100:
            cost = 100
        return cost

    def find_closest_point(self, robot_position, contours, costmap_matrix, max_cost=30.0):
        cost = max_cost
        closest_point = None
        self.get_logger().debug(f'Costmap matrix shape: {costmap_matrix.shape}')
        for contour in contours:
            for point in contour:
                px, py = point[0]
                dist = np.hypot(px - robot_position[0], py - robot_position[1])
                new_cost = self.get_cost_heuristic(costmap_matrix, px, py, robot_position, dist)
                self.get_logger().debug(f'Cost at {px}, {py}: {new_cost}')
                if new_cost < cost:
                    cost = new_cost
                    closest_point = (px, py)
        if closest_point:
            self.get_logger().info(f'Closest point: {closest_point}')
            self.get_logger().info(f'Cost: {cost}')
            x, y = closest_point
            self.get_logger().info(f'costmap arround: {costmap_matrix[y-3:y+4, x-3:x+4]}')
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
        if self.no_valid_goal_count > 2:
            self.get_logger().info('No valid goal found for too long. Environment may be fully explored')
            raise SystemExit

        binary_known_cells_map = self.get_known_map_binary_image()
        costmap_matrix = self.get_costmap_matrix()
        contours, _ = cv2.findContours(binary_known_cells_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        try:
            t = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0))
        except TransformException as ex:
            self.get_logger().info(f'Could not transform map to base_link: {ex}')
            return

        robot_position = self.get_robot_position_map_image_frame(t.transform.translation.x, t.transform.translation.y)
        closest_point = self.find_closest_point(robot_position, contours, costmap_matrix)
        if self.debug_cv:
            self.publish_contour_image(binary_known_cells_map, contours, robot_position, closest_point)

        if closest_point is None:
            self.get_logger().info('No valid goal found')
            self.no_valid_goal_count += 1
            return

        goal_pose = self.get_goal_pose(closest_point)
        self.goal_pose_pub_.publish(goal_pose)
        self.get_logger().info(f'Published goal at: {goal_pose.pose.position.x}, {goal_pose.pose.position.y}')

        self.no_valid_goal_count = 0

    def process_map(self):
        if not self.current_map or not self.current_costmap:
            self.get_logger().info('Map(s) not received yet')
            return
        self.get_logger().info('Processing map')
        self.find_goal()

def main(args=None):
    rclpy.init(args=args)
    node = Pilot()
    try:
        rclpy.spin(node)
    except SystemExit:
        rclpy.logging.get_logger('pilot').info('Exiting')
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
