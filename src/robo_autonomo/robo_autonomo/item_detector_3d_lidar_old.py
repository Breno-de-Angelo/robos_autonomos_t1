import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
import tf2_ros
import geometry_msgs.msg
import numpy as np
import open3d as o3d
import math
import atexit

def pointcloud2_to_xyz(msg):
    """
    Converts a sensor_msgs/PointCloud2 message to a (N x 3) numpy array.
    """
    points_list = []
    for point in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
        points_list.append([point[0], point[1], point[2]])
    return np.array(points_list, dtype=np.float32)

class ObjectDetector(Node):
    def __init__(self):
        super().__init__('object_detector')
        # Declare a parameter to enable debug mode
        self.declare_parameter('debug', False)
        self.debug = self.get_parameter('debug').value

        # Subscribe to the 3D lidar point cloud
        self.pc_sub = self.create_subscription(
            PointCloud2,
            '/velodyne2/velodyne_points2',  # Adjust this topic name as needed
            self.pc_callback,
            10)
        
        # Timer to process point clouds periodically
        self.timer = self.create_timer(1.0, self.detect_items)
    
        self.latest_pc_msg = None
        self.detections = []  # List to store detected objects

        # TF broadcaster for detected objects
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # If debug is enabled, publish a debug point cloud
        if self.debug:
            self.debug_pub = self.create_publisher(PointCloud2, '/debug_item_detection', 10)
            self.get_logger().info("Debug mode enabled; publishing to /debug_item_detection.")

        # Ensure that shutdown_callback is called on exit
        atexit.register(self.shutdown_callback)

    def pc_callback(self, msg: PointCloud2):
        self.latest_pc_msg = msg

    def detect_items(self):
        if self.latest_pc_msg is None:
            self.get_logger().warn("No point cloud received yet.")
            return

        # Convert the ROS PointCloud2 message to a numpy array
        points = pointcloud2_to_xyz(self.latest_pc_msg)
        if points.size == 0:
            self.get_logger().warn("Empty point cloud.")
            return

        # Create an Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Make a copy for debug visualization (if enabled)
        debug_pcd = pcd.clone() if self.debug else None

        # --- Sphere detection using RANSAC ---
        # We work on the full point cloud first.
        remaining_points = np.asarray(pcd.points)
        sphere_detections = []
        # Try to detect one sphere (you could extend this to multiple iterations)
        sphere_model, inliers = self.detect_sphere(
            remaining_points,
            num_iterations=1000,
            threshold=0.05,     # distance threshold (meters) – adjust as needed
            min_inliers=50      # minimum number of inliers to accept a sphere
        )
        if sphere_model is not None:
            center, radius = sphere_model
            sphere_detections.append((sphere_model, inliers))
            self.get_logger().info(f"Detected sphere: center={center}, radius={radius:.2f}")
            # Remove sphere inliers from the remaining points for further processing
            remaining_points = np.delete(remaining_points, inliers, axis=0)

        # --- Box detection via clustering and bounding boxes ---
        box_detections = []
        if remaining_points.shape[0] > 0:
            pcd_remaining = o3d.geometry.PointCloud()
            pcd_remaining.points = o3d.utility.Vector3dVector(remaining_points)
            # Cluster the remaining points with DBSCAN
            labels = np.array(pcd_remaining.cluster_dbscan(eps=0.2, min_points=30, print_progress=False))
            max_label = int(labels.max()) if labels.size > 0 else -1
            for i in range(max_label + 1):
                cluster_indices = np.where(labels == i)[0]
                if cluster_indices.size < 30:
                    continue
                cluster_points = remaining_points[cluster_indices]
                cluster_pcd = o3d.geometry.PointCloud()
                cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points)
                # Compute an oriented bounding box around the cluster
                obb = cluster_pcd.get_oriented_bounding_box()
                box_detections.append((obb, cluster_points))
                self.get_logger().info(f"Detected box: center={obb.center}, dimensions={obb.extent}")

        # --- Broadcast detected objects as TF and store detections ---
        for sphere, inliers in sphere_detections:
            center, radius = sphere
            frame_id = f"sphere_{len(self.detections)}"
            self.broadcast_tf(center, frame_id)
            self.detections.append({'type': 'sphere', 'position': center, 'radius': radius})
        for obb, cluster_points in box_detections:
            center = obb.center
            frame_id = f"box_{len(self.detections)}"
            self.broadcast_tf(center, frame_id)
            self.detections.append({'type': 'box', 'position': center, 'dimensions': obb.extent})

        # --- Publish debug point cloud (colored) if debug mode is enabled ---
        if self.debug and debug_pcd is not None:
            # Color all points gray by default
            pts = np.asarray(debug_pcd.points)
            colors = np.tile(np.array([[0.5, 0.5, 0.5]]), (pts.shape[0], 1))
            # Color sphere inliers green
            for sphere, inliers in sphere_detections:
                colors[inliers] = [0, 1, 0]
            # For box clusters, we run DBSCAN on the full cloud (to recover indices)
            labels = np.array(pcd.cluster_dbscan(eps=0.2, min_points=30, print_progress=False))
            for i in range(int(labels.max()) + 1):
                cluster_indices = np.where(labels == i)[0]
                # Check if this cluster's centroid is near any detected box center
                cluster_pts = np.asarray(pcd.points)[cluster_indices]
                cluster_center = np.mean(cluster_pts, axis=0)
                for obb, _ in box_detections:
                    if np.linalg.norm(cluster_center - obb.center) < 0.2:
                        colors[cluster_indices] = [0, 0, 1]
                        break
            debug_pcd.colors = o3d.utility.Vector3dVector(colors)
            debug_msg = self.convert_open3d_to_ros(debug_pcd)
            self.debug_pub.publish(debug_msg)

    def detect_sphere(self, points, num_iterations=1000, threshold=0.05, min_inliers=50):
        """
        Runs RANSAC to detect a sphere in the given Nx3 array of points.
        Returns a tuple ( (center, radius), inlier_indices ) if a valid sphere is found;
        otherwise, returns (None, []).
        """
        best_inliers = []
        best_model = None
        n_points = points.shape[0]
        if n_points < 4:
            return None, []
        for i in range(num_iterations):
            indices = np.random.choice(n_points, 4, replace=False)
            sample = points[indices]
            A = []
            B = []
            for pt in sample:
                x, y, z = pt
                A.append([-2 * x, -2 * y, -2 * z, 1])
                B.append(-(x**2 + y**2 + z**2))
            A = np.array(A)
            B = np.array(B)
            try:
                sol, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)
            except np.linalg.LinAlgError:
                continue
            a, b, c, D = sol
            center = np.array([a, b, c])
            radius_sq = a * a + b * b + c * c - D
            if radius_sq < 0:
                continue
            radius = math.sqrt(radius_sq)
            dists = np.linalg.norm(points - center, axis=1)
            residuals = np.abs(dists - radius)
            inlier_indices = np.where(residuals < threshold)[0]
            if len(inlier_indices) > len(best_inliers):
                best_inliers = inlier_indices
                best_model = (center, radius)
            if len(best_inliers) > min_inliers:
                break
        if best_model is not None and len(best_inliers) >= min_inliers:
            return best_model, best_inliers
        else:
            return None, []

    def broadcast_tf(self, position, object_frame_id):
        """
        Broadcasts a TF transform for the detected object.
        The parent frame is assumed to be "base_link".
        """
        t = geometry_msgs.msg.TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "base_link"
        t.child_frame_id = object_frame_id

        t.transform.translation.x = float(position[0])
        t.transform.translation.y = float(position[1])
        t.transform.translation.z = float(position[2])
        # Use an identity rotation (no orientation)
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform(t)
        self.get_logger().info(f"Broadcasted TF for {object_frame_id} at {position}")

    def convert_open3d_to_ros(self, pcd, frame_id="base_link"):
        """
        Converts an Open3D PointCloud to a sensor_msgs/PointCloud2 message.
        """
        points = np.asarray(pcd.points)
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = frame_id
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
        ]
        cloud_msg = pc2.create_cloud(header, fields, points)
        return cloud_msg

    def shutdown_callback(self):
        """
        Writes all detected objects (with type and parameters) to a text file on shutdown.
        """
        filename = "detections.txt"
        try:
            with open(filename, 'w') as f:
                for det in self.detections:
                    if det['type'] == 'sphere':
                        center = det['position']
                        radius = det['radius']
                        f.write(f"sphere: center=({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}), radius={radius:.2f}\n")
                    elif det['type'] == 'box':
                        center = det['position']
                        dims = det['dimensions']
                        f.write(f"box: center=({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}), dimensions=({dims[0]:.2f}, {dims[1]:.2f}, {dims[2]:.2f})\n")
            self.get_logger().info(f"Detections saved to {filename}")
        except Exception as e:
            self.get_logger().error(f"Failed to write detections to file: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown_callback()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
