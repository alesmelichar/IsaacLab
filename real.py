import urx
import time
import numpy as np

robot_ip = "147.175.108.138"
#robot_ip = "192.168.56.1"
#port = 50002

class SafeRobotTester:
    def __init__(self, ip):
        self.ip = ip
        self.robot = None
        self.home_position = None
        
    def connect(self):
        try:
            self.robot = urx.Robot(self.ip, use_rt=True)
            print("✅ Connected successfully!")
            self.home_position = self.robot.getj()
            print(f"Home position recorded: {self.home_position}")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    def disconnect(self):
        """Close connection safely"""
        if self.robot:
            self.robot.close()
            print("🔌 Disconnected safely")
    
    def get_current_state(self):
        """Get current robot state"""
        joints = self.robot.getj()
        pose = self.robot.get_pose()
        return {
            'joints': joints,
            'pose': pose,
            'joint_speeds': self.robot.getj_vel() if hasattr(self.robot, 'getj_vel') else None
        }

    def move_to_home(self):
        """Return to recorded home position"""
        if self.home_position:
            print("Returning to home position...")
            self.robot.movej(self.home_position, acc=0.1, vel=0.1)
            print("✅ Returned home")

    def test_wrist3_rotation(self, delta=0.087):
        """
        Test a small rotation on wrist3 (joint 5).
        :param delta: angle in radians (default ~5 degrees)
        """
        if not self.robot:
            print("❌ Robot is not connected!")
            return
        
        # Get current joint positions
        current_joints = self.robot.getj()
        print(f"Current joints: {np.round(current_joints, 3)}")

        # Prepare new target
        target_joints = current_joints.copy()
        target_joints[5] += delta  # Wrist3 = joint index 5
        
        print(f"Target joints: {np.round(target_joints, 3)}")

        # Move with slow speed for safety
        self.robot.movej(target_joints, acc=0.05, vel=0.05, wait=True)
        print("✅ Wrist3 rotated successfully!")

# -------------------- RUN TEST --------------------
tester = SafeRobotTester(robot_ip)

if tester.connect():
    state = tester.get_current_state()
    print(f"Current state: {state}")
    tester.test_wrist3_rotation(delta=np.deg2rad(1))
    time.sleep(1)
    tester.move_to_home()
    tester.disconnect()
