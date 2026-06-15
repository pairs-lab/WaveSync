import time
import pybullet as pb
import pybullet_data

try:
    from utils.RobotConfig import RobotConfig
    from utils.types import RobotType
except ImportError:
    # print("Warning: Could not import RobotConfig from utils.")
    pass

class PyBulletEnv:
    def __init__(self, robot_type=None, video_out=None):
        self.robot_id, self.joint_name_to_id = self.setup_pybullet(robot_type, video_out)
        self.neutral_pose = {
            j_name: pb.getJointState(self.robot_id, j_id)[0] 
            for j_name, j_id in self.joint_name_to_id.items()
        }
        self.current_pose = self.neutral_pose.copy()

    def setup_pybullet(self, robot_type, video_out=None):
        # print("\n[PyBullet] Initializing environment...")
        pb.connect(pb.GUI)
        pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
        
        if video_out:
            self.log_id = pb.startStateLogging(pb.STATE_LOGGING_VIDEO_MP4, video_out)
            
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        pb.setGravity(0, 0, -11.7)
        pb.loadURDF("plane.urdf")
        
        try:
            r_type = robot_type if robot_type else RobotType.COMAN
            robot_config = RobotConfig(r_type)
            robot_id = pb.loadURDF(robot_config.URDF_4_RENDER_PATH)
        except Exception as e:
            # print(f"Error loading URDF: {e}. Using r2d2 temporarily.")
            robot_id = pb.loadURDF("r2d2.urdf", [0, 0, 0.5])
            
        pb.changeDynamics(robot_id, -1, mass=0)
        initial_position = [0, 0, 0.53]
        initial_orientation = pb.getQuaternionFromEuler([0, 0, 0])
        pb.resetBasePositionAndOrientation(robot_id, initial_position, initial_orientation)
        pb.resetDebugVisualizerCamera(
            cameraDistance=1.2, cameraYaw=90, cameraPitch=-15, cameraTargetPosition=initial_position
        )
        
        num_joints = pb.getNumJoints(robot_id)
        joint_name_to_id = {
            pb.getJointInfo(robot_id, i)[1].decode("utf-8"): pb.getJointInfo(robot_id, i)[0] 
            for i in range(num_joints)
        }
        return robot_id, joint_name_to_id

    def play_frames(self, frames, start_t=None, max_duration=None, fps=30):
        """
        Plays a sequence of frames in the simulation.
        If start_t and max_duration are provided, it breaks if time exceeds max_duration.
        """
        frame_time = 1.0 / fps
        for joi_angles in frames:
            loop_start = time.time()
            for joi_name, angle in joi_angles.items():
                if joi_name in self.joint_name_to_id:
                    joi_index = self.joint_name_to_id[joi_name]
                    pb.resetJointState(self.robot_id, joi_index, angle)
                    self.current_pose[joi_name] = angle
            pb.stepSimulation()
            
            elapsed = time.time() - loop_start
            if frame_time > elapsed: 
                time.sleep(frame_time - elapsed)
            
            if start_t and max_duration:
                if (time.time() - start_t) > max_duration: 
                    break

    def disconnect(self):
        if hasattr(self, 'log_id'):
            pb.stopStateLogging(self.log_id)
        pb.disconnect()
