# Requirements Document: YOLO Person Detector Pipeline

## 1. Functional Requirements

### 1.1 Camera Management

**1.1.1** The system SHALL support video input from all four Agibot X2 camera sensors:
- RGB Head Center Camera
- RGB Head Rear Camera
- RGBD Head Front Camera
- Stereo Head Front Camera (left sensor)

**1.1.2** The system SHALL provide dynamic camera selection capability without requiring pipeline restart.

**1.1.3** The system SHALL subscribe to standard ROS2 camera topics publishing `sensor_msgs/Image` messages.

**1.1.4** The system SHALL publish the selected camera stream to a unified output topic at the target frame rate.

**1.1.5** The system SHALL perform image preprocessing including resize and format conversion to match YOLO input requirements.

**1.1.6** The system SHALL expose camera selection via ROS2 parameter and service interface.

### 1.2 Person Detection

**1.2.1** The system SHALL load and initialize a YOLO object detection model (YOLOv8 or YOLOv10).

**1.2.2** The system SHALL process incoming image frames through the YOLO model to detect persons.

**1.2.3** The system SHALL filter detections to include only the "person" class (class_id = 0).

**1.2.4** The system SHALL apply configurable confidence threshold filtering to remove low-confidence detections.

**1.2.5** The system SHALL perform Non-Maximum Suppression (NMS) to eliminate duplicate detections.

**1.2.6** The system SHALL publish detection results as `vision_msgs/Detection2DArray` messages.

**1.2.7** The system SHALL support both GPU (CUDA) and CPU inference modes.

**1.2.8** The system SHALL provide model reload capability via ROS2 service without node restart.

### 1.3 3D Spatial Processing

**1.3.1** The system SHALL convert 2D bounding boxes to 3D positions when depth data is available.

**1.3.2** The system SHALL use depth information from RGBD camera to calculate person distance from robot.

**1.3.3** The system SHALL transform detection coordinates from camera frame to robot base frame using TF2.

**1.3.4** The system SHALL calculate person bearing angle relative to robot heading.

**1.3.5** The system SHALL filter detections outside the valid depth range (0.3m - 10.0m).

**1.3.6** The system SHALL publish enriched detections with 3D spatial information.

**1.3.7** The system SHALL publish person poses as `geometry_msgs/PoseArray` messages.

### 1.4 Visualization

**1.4.1** The system SHALL overlay bounding boxes on camera images for detected persons.

**1.4.2** The system SHALL display confidence scores on visualization images.

**1.4.3** The system SHALL display person distance on visualization images when depth is available.

**1.4.4** The system SHALL publish annotated images for WebRTC streaming and debugging.

**1.4.5** The system SHALL generate 3D markers for RViz visualization.

**1.4.6** The system SHALL provide configurable visualization parameters (colors, thickness, font size).

### 1.5 ROS2 Integration

**1.5.1** The system SHALL integrate with existing ROS2 Humble infrastructure on Agibot X2.

**1.5.2** The system SHALL use the robot's TF2 transform tree for coordinate transformations.

**1.5.3** The system SHALL publish detection results compatible with rosbridge WebSocket server.

**1.5.4** The system SHALL publish visualization images compatible with WebRTC video server.

**1.5.5** The system SHALL follow ROS2 naming conventions and topic hierarchy.

**1.5.6** The system SHALL use appropriate QoS profiles for different message types.

## 2. Performance Requirements

### 2.1 Real-Time Processing

**2.1.1** The system SHALL process video frames at a minimum rate of 10 Hz.

**2.1.2** The system SHOULD process video frames at a target rate of 30 Hz when using GPU acceleration.

**2.1.3** The system SHALL maintain inference time below 33.3ms per frame when using YOLOv8n on GPU.

**2.1.4** The system SHALL maintain total pipeline latency below 50ms from camera input to detection output.

**2.1.5** The system SHALL not drop frames under normal operating conditions with single camera input.

### 2.2 Resource Utilization

**2.2.1** The system SHALL operate within 50 MB of RAM for model and buffers.

**2.2.2** The system SHALL use no more than 500 MB of GPU VRAM when using GPU acceleration.

**2.2.3** The system SHALL use no more than 80% of a single CPU core when running in CPU inference mode.

**2.2.4** The system SHALL use no more than 20% GPU utilization when running in GPU inference mode.

### 2.3 Scalability

**2.3.1** The system SHALL support sequential processing of multiple camera streams.

**2.3.2** The system SHALL handle up to 10 simultaneous person detections per frame without performance degradation.

**2.3.3** The system SHALL maintain stable performance during continuous operation for at least 1 hour.

## 3. Reliability Requirements

### 3.1 Error Handling

**3.1.1** The system SHALL detect camera stream timeout (no images for > 1 second) and publish diagnostic warning.

**3.1.2** The system SHALL attempt automatic reconnection to camera topic after timeout.

**3.1.3** The system SHALL gracefully handle model inference failures by skipping the failed frame and continuing.

**3.1.4** The system SHALL restart the detector node if more than 10 consecutive inference failures occur.

**3.1.5** The system SHALL handle invalid depth data by publishing 2D-only detections without 3D position.

**3.1.6** The system SHALL buffer detections for up to 100ms when TF transform is temporarily unavailable.

**3.1.7** The system SHALL fallback to CPU inference if GPU out-of-memory error occurs.

### 3.2 Robustness

**3.2.1** The system SHALL validate model file existence and integrity at startup.

**3.2.2** The system SHALL validate all bounding box coordinates are within image boundaries.

**3.2.3** The system SHALL validate all confidence scores are in the range [0.0, 1.0].

**3.2.4** The system SHALL validate depth values are finite and within valid range before 3D projection.

**3.2.5** The system SHALL handle camera disconnection without crashing.

**3.2.6** The system SHALL recover automatically when camera stream is restored.

### 3.3 Diagnostics

**3.3.1** The system SHALL publish diagnostic messages for all error conditions.

**3.3.2** The system SHALL log inference time statistics at 1 Hz.

**3.3.3** The system SHALL log camera switching events.

**3.3.4** The system SHALL provide node health status via ROS2 diagnostics.

## 4. Accuracy Requirements

### 4.1 Detection Quality

**4.1.1** The system SHALL only publish detections with confidence scores above the configured threshold (default 0.5).

**4.1.2** The system SHALL apply NMS with IoU threshold of 0.45 to eliminate duplicate detections.

**4.1.3** The system SHALL correctly identify the "person" class using the YOLO model's class predictions.

**4.1.4** The system SHALL maintain detection accuracy consistent with the selected YOLO model's published performance metrics.

### 4.2 Spatial Accuracy

**4.2.1** The system SHALL calculate 3D positions with accuracy limited by depth sensor specifications (typically ±2% at 2m).

**4.2.2** The system SHALL apply camera intrinsic calibration parameters for accurate 3D projection.

**4.2.3** The system SHALL use TF2 transforms with timestamp synchronization within 10ms.

**4.2.4** The system SHALL calculate person distance with error less than 5% of true distance within valid range.

## 5. Usability Requirements

### 5.1 Configuration

**5.1.1** The system SHALL provide YAML configuration files for all adjustable parameters.

**5.1.2** The system SHALL support runtime parameter updates via ROS2 parameter services.

**5.1.3** The system SHALL provide sensible default values for all configuration parameters.

**5.1.4** The system SHALL validate configuration parameters at startup and reject invalid values.

### 5.2 Deployment

**5.2.1** The system SHALL provide ROS2 launch files for common deployment scenarios.

**5.2.2** The system SHALL support single-command launch of the complete detection pipeline.

**5.2.3** The system SHALL provide clear error messages for missing dependencies or configuration issues.

**5.2.4** The system SHALL include installation instructions and dependency list in documentation.

### 5.3 Monitoring

**5.3.1** The system SHALL provide RViz configuration for 3D visualization of detections.

**5.3.2** The system SHALL publish annotated images viewable in standard ROS2 image viewers.

**5.3.3** The system SHALL expose detection data via rosbridge for web-based monitoring.

**5.3.4** The system SHALL provide inference time metrics for performance monitoring.

## 6. Interface Requirements

### 6.1 Input Interfaces

**6.1.1** The system SHALL subscribe to camera image topics with message type `sensor_msgs/Image`.

**6.1.2** The system SHALL subscribe to camera info topics with message type `sensor_msgs/CameraInfo`.

**6.1.3** The system SHALL subscribe to depth image topics with message type `sensor_msgs/Image` (for RGBD camera).

**6.1.4** The system SHALL use TF2 for camera-to-base coordinate transforms.

### 6.2 Output Interfaces

**6.2.1** The system SHALL publish detections on `/yolo/detections` topic with message type `vision_msgs/Detection2DArray`.

**6.2.2** The system SHALL publish enriched detections on `/yolo/person_detections` topic with custom message type `PersonDetectionArray`.

**6.2.3** The system SHALL publish person poses on `/yolo/person_poses` topic with message type `geometry_msgs/PoseArray`.

**6.2.4** The system SHALL publish visualization markers on `/yolo/person_markers` topic with message type `visualization_msgs/MarkerArray`.

**6.2.5** The system SHALL publish annotated images on `/yolo/visualization_image` topic with message type `sensor_msgs/Image`.

**6.2.6** The system SHALL publish inference time on `/yolo/inference_time` topic with message type `std_msgs/Float32`.

### 6.3 Service Interfaces

**6.3.1** The system SHALL provide camera selection service `/yolo/select_camera` with type `std_srvs/SetString`.

**6.3.2** The system SHALL provide model reload service `/yolo/reload_model` with type `std_srvs/Trigger`.

### 6.4 Parameter Interfaces

**6.4.1** The system SHALL expose all configuration parameters via ROS2 parameter server.

**6.4.2** The system SHALL support dynamic reconfiguration of runtime parameters.

**6.4.3** The system SHALL validate parameter values and reject invalid updates.

## 7. Data Requirements

### 7.1 Message Definitions

**7.1.1** The system SHALL define custom message type `PersonDetection` containing:
- Header with timestamp and frame_id
- 2D detection information (bounding box, confidence)
- 3D spatial information (position, distance, bearing)
- Metadata (camera source, person ID)

**7.1.2** The system SHALL define custom message type `PersonDetectionArray` containing:
- Header with timestamp and frame_id
- Array of PersonDetection messages
- Number of detected persons
- Inference time in milliseconds

### 7.2 Data Validation

**7.2.1** The system SHALL validate all bounding box coordinates are non-negative integers.

**7.2.2** The system SHALL validate bounding box dimensions are positive integers.

**7.2.3** The system SHALL validate confidence scores are in range [0.0, 1.0].

**7.2.4** The system SHALL validate depth values are finite positive numbers within valid range.

**7.2.5** The system SHALL validate timestamps are monotonically increasing.

### 7.3 Data Synchronization

**7.3.1** The system SHALL synchronize detection timestamps with input image timestamps.

**7.3.2** The system SHALL synchronize depth data with RGB image data when using RGBD camera.

**7.3.3** The system SHALL use TF2 transform timestamps matching detection timestamps.

## 8. Security Requirements

### 8.1 Data Privacy

**8.1.1** The system SHALL NOT store camera images or detection data persistently by default.

**8.1.2** The system SHALL provide option to disable data logging for privacy-sensitive deployments.

**8.1.3** The system SHALL document potential privacy implications of person detection in user documentation.

### 8.2 Network Security

**8.2.1** The system SHALL support ROS2 DDS security when enabled on the robot.

**8.2.2** The system SHALL be compatible with SROS2 (Secure ROS2) authentication and encryption.

**8.2.3** The system SHALL support secure WebSocket connections (WSS) for rosbridge integration.

### 8.3 Model Security

**8.3.1** The system SHALL verify model file integrity using checksums at load time.

**8.3.2** The system SHALL reject model files that fail integrity verification.

**8.3.3** The system SHALL log model loading events for audit purposes.

## 9. Compatibility Requirements

### 9.1 Platform Compatibility

**9.1.1** The system SHALL run on Ubuntu 22.04 LTS.

**9.1.2** The system SHALL be compatible with ROS2 Humble distribution.

**9.1.3** The system SHALL support Python 3.10 or later.

**9.1.4** The system SHALL support both x86_64 and ARM64 architectures.

### 9.2 Hardware Compatibility

**9.2.1** The system SHALL run on Agibot X2 humanoid robot hardware.

**9.2.2** The system SHALL support NVIDIA GPUs with CUDA 11.8 or later for GPU acceleration.

**9.2.3** The system SHALL provide CPU-only inference mode for systems without GPU.

**9.2.4** The system SHALL be compatible with standard USB cameras and RealSense depth cameras.

### 9.3 Software Compatibility

**9.3.1** The system SHALL be compatible with existing robot state publisher on Agibot X2.

**9.3.2** The system SHALL be compatible with rosbridge WebSocket server.

**9.3.3** The system SHALL be compatible with WebRTC video server.

**9.3.4** The system SHALL use standard ROS2 message types where possible.

## 10. Maintainability Requirements

### 10.1 Code Quality

**10.1.1** The system SHALL follow PEP 8 Python coding standards.

**10.1.2** The system SHALL include docstrings for all public functions and classes.

**10.1.3** The system SHALL maintain code coverage above 80% for unit tests.

**10.1.4** The system SHALL use type hints for function signatures.

### 10.2 Documentation

**10.2.1** The system SHALL include README with installation and usage instructions.

**10.2.2** The system SHALL document all ROS2 topics, services, and parameters.

**10.2.3** The system SHALL provide example launch files with inline comments.

**10.2.4** The system SHALL document all configuration parameters with descriptions and valid ranges.

### 10.3 Testing

**10.3.1** The system SHALL include unit tests for all core components.

**10.3.2** The system SHALL include integration tests for end-to-end pipeline.

**10.3.3** The system SHALL include property-based tests for critical algorithms.

**10.3.4** The system SHALL provide test data (recorded bag files) for regression testing.

### 10.4 Modularity

**10.4.1** The system SHALL implement each major component as a separate ROS2 node.

**10.4.2** The system SHALL use well-defined interfaces between components.

**10.4.3** The system SHALL allow individual components to be replaced or upgraded independently.

**10.4.4** The system SHALL separate YOLO model interface from node implementation for easy model swapping.

## 11. Extensibility Requirements

### 11.1 Model Flexibility

**11.1.1** The system SHALL support loading different YOLO model variants (YOLOv8n, YOLOv8s, YOLOv8m, YOLOv10).

**11.1.2** The system SHALL provide abstraction layer for YOLO model interface to support future model architectures.

**11.1.3** The system SHALL support ONNX model format for cross-platform deployment.

**11.1.4** The system SHALL support TensorRT optimized models for maximum GPU performance.

### 11.2 Feature Extensions

**11.2.1** The system architecture SHALL allow addition of person tracking module without major refactoring.

**11.2.2** The system architecture SHALL allow addition of multi-camera fusion module.

**11.2.3** The system architecture SHALL allow addition of pose estimation module.

**11.2.4** The system architecture SHALL allow addition of action recognition module.

### 11.3 Camera Extensions

**11.3.1** The system SHALL support addition of new camera sources without code changes to detector node.

**11.3.2** The system SHALL support different camera resolutions and aspect ratios.

**11.3.3** The system SHALL support different depth sensor types (stereo, structured light, ToF).

## 12. Deployment Requirements

### 12.1 Installation

**12.1.1** The system SHALL provide installation script for all dependencies.

**12.1.2** The system SHALL be installable as a standard ROS2 package using colcon build.

**12.1.3** The system SHALL include pre-trained YOLO model weights or download script.

**12.1.4** The system SHALL verify all dependencies are installed at first launch.

### 12.2 Configuration Management

**12.2.1** The system SHALL store configuration files in standard ROS2 config directory.

**12.2.2** The system SHALL support environment-specific configuration files (dev, test, prod).

**12.2.3** The system SHALL provide configuration validation tool.

**12.2.4** The system SHALL document all configuration options with examples.

### 12.3 Startup and Shutdown

**12.3.1** The system SHALL start all nodes in correct dependency order via launch file.

**12.3.2** The system SHALL perform graceful shutdown when receiving SIGINT or SIGTERM.

**12.3.3** The system SHALL clean up resources (GPU memory, file handles) on shutdown.

**12.3.4** The system SHALL complete startup within 10 seconds on target hardware.

## 13. Compliance Requirements

### 13.1 Standards Compliance

**13.1.1** The system SHALL comply with ROS2 Humble API specifications.

**13.1.2** The system SHALL follow ROS2 naming conventions (REP-144).

**13.1.3** The system SHALL use standard ROS2 message types where applicable (REP-145).

**13.1.4** The system SHALL follow ROS2 QoS best practices.

### 13.2 License Compliance

**13.2.1** The system SHALL use only open-source dependencies with compatible licenses.

**13.2.2** The system SHALL document all third-party licenses in LICENSE file.

**13.2.3** The system SHALL comply with YOLO model license terms (AGPL-3.0 for YOLOv8).

**13.2.4** The system SHALL provide clear attribution for all third-party components.

## 14. Constraints

### 14.1 Technical Constraints

**14.1.1** The system MUST use ROS2 Humble (no other ROS2 distributions supported).

**14.1.2** The system MUST use Python 3.10+ (no Python 2.x support).

**14.1.3** The system MUST use Ultralytics YOLO implementation (no custom YOLO implementations).

**14.1.4** The system MUST operate within robot's computational resources (no external compute servers).

### 14.2 Operational Constraints

**14.2.1** The system MUST NOT interfere with robot's primary control systems.

**14.2.2** The system MUST NOT consume more than 30% of total robot CPU resources.

**14.2.3** The system MUST NOT block robot's main control loop.

**14.2.4** The system MUST operate in real-time without buffering more than 3 frames.

### 14.3 Environmental Constraints

**14.3.1** The system MUST operate in indoor lighting conditions (100-1000 lux).

**14.3.2** The system MUST handle varying lighting conditions without manual adjustment.

**14.3.3** The system MUST operate with persons at distances from 0.3m to 10m.

**14.3.4** The system MUST handle partial occlusions and multiple persons in frame.

## 15. Assumptions and Dependencies

### 15.1 Assumptions

**15.1.1** Camera drivers are properly installed and configured on the robot.

**15.1.2** Camera calibration parameters are accurate and available via CameraInfo topics.

**15.1.3** Robot state publisher is running and publishing TF transforms.

**15.1.4** Network connectivity is available for rosbridge and WebRTC (if used).

**15.1.5** YOLO model weights are pre-trained on COCO dataset or equivalent.

### 15.2 External Dependencies

**15.2.1** The system depends on ROS2 Humble being installed and sourced.

**15.2.2** The system depends on camera drivers publishing standard sensor_msgs/Image messages.

**15.2.3** The system depends on TF2 tree being published by robot state publisher.

**15.2.4** The system depends on CUDA and cuDNN for GPU acceleration (optional).

**15.2.5** The system depends on rosbridge and WebRTC servers for remote visualization (optional).

