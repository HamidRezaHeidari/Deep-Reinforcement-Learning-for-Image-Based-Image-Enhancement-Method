# Machine Vision Final Project - Dr. Shariatmadar - Author: Hamid Reza Heidari / Milad Mohammadi

# import library
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from ultralytics import YOLO
import cv2

from customtkinter import *
from PIL import Image

n_state = 64
n_action = 9

# create CNN
class FeatureExtractorCNN(nn.Module):
    def __init__(self, output_dim=64):
        super(FeatureExtractorCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(32 * 4 * 4, output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        features = self.fc(x)
        return features

# create Q-Network
class QNetwork(nn.Module):

    def __init__(self, state_size=n_state, action_size=n_action, hidden_size=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x1 = F.relu(self.fc1(state))
        x2 = F.relu(self.fc2(x1))
        x3 = F.relu(self.fc3(x2))
        return self.fc4(x3)


def image_enhance_action(idx, image, cl, tgs, bs, ms, ks):

    if idx == 0 or idx == 1 or idx == 2 or idx == 3:
        # Robot Should Move to its side and capture new picture
        return image

    # CLAHE
    elif idx == 4:
        #HP
        clip_limit = cl
        tile_grid_size = (tgs, tgs)

        # Convert to LAB color space
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # Extract L (lightness) channel
        l_channel, a_channel, b_channel = cv2.split(lab_image)

        # Apply CLAHE to the L channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l_channel_clahe = clahe.apply(l_channel)

        # Merge modified L channel back with a and b channels
        lab_image_clahe = cv2.merge((l_channel_clahe, a_channel, b_channel))

        # Convert back to BGR color space
        image_clahe = cv2.cvtColor(lab_image_clahe, cv2.COLOR_LAB2BGR)
        return image_clahe

    # Saturation HSV
    elif idx == 5:
        #HP
        base_scale = bs
        max_scale = ms

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Split the HSV channels
        h, s, v = cv2.split(hsv_image)

        # Compute statistics of the saturation channel
        mean_saturation = np.mean(s)

        # Calculate the adaptive scaling factor
        # If the mean saturation is low, apply a stronger adjustment
        if mean_saturation < 25:  # Low saturation
            saturation_scale = max_scale
        elif mean_saturation < 55:  # Medium saturation
            saturation_scale = base_scale + ((max_scale - base_scale) * (170 - mean_saturation) / 85)
        else:  # High saturation
            saturation_scale = base_scale

        # Scale the saturation channel
        s = np.clip(s * saturation_scale, 0, 255).astype(np.uint8)

        # Merge the channels back
        enhanced_hsv = cv2.merge((h, s, v))

        # Convert back to BGR color space
        enhanced_image = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)

        return enhanced_image

    # denoise gaussian
    elif idx == 6:
        kernel_size = ks

        # Apply bilateral filter
        denoised_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return denoised_image

    # Histogram EQu
    elif idx == 7:

        # RGB to YUV
        yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

        # Apply Equalization Hist to Y Channel
        yuv_image[:, :, 0] = cv2.equalizeHist(yuv_image[:, :, 0])

        # Image to RGB
        enhanced_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)

        return enhanced_image

    # do nothing
    elif idx == 8:
        return image

    else:
        print("Error 102 arise >>>>")



############################################## Webots Controller on UR5e
import sys
webot_path = 'C:\Program Files\Webots\lib\controller\python'
sys.path.append(webot_path)

from controller import Robot, VacuumGripper, Motor, Camera, RangeFinder

# create the Robot instance
robot = Robot()

TIME_STEP = 64

camera = Camera('camera')
range_finder = RangeFinder('range-finder')

range_finder.enable(TIME_STEP)
camera.enable(TIME_STEP)


## grab RGB frame and convert to numpy ndarray
def get_rgb_frame() -> np.ndarray:
    image_array = camera.getImageArray()
    np_image = np.array(image_array, dtype=np.uint8).reshape((camera.getHeight(), camera.getWidth(), 3))
    return np_image


# grab Depth frame and convert to numpy ndarray
def get_depth_frame() -> np.ndarray:
    image_array = range_finder.getImageArray()
    np_image = np.array(image_array, dtype=np.uint8).reshape((camera.getHeight(), camera.getWidth(), 3))
    return image_array

class UR5e:
    def __init__(self, name="my_robot"):
        # get the motor devices
        m1 = robot.getDevice('shoulder_lift_joint')
        m2 = robot.getDevice('shoulder_pan_joint')
        m3 = robot.getDevice('elbow_joint')

        m4 = robot.getDevice('wrist_1_joint')
        m5 = robot.getDevice('wrist_2_joint')
        m6 = robot.getDevice('wrist_3_joint')

        self.vac = robot.getDevice('vacuum gripper')
        self.vac.enablePresence(1)

        self.motors_list = [m1, m2, m3, m4, m5, m6]

        self.gps = robot.getDevice('gps')
        self.gps.enable(1)

        sampling_period = 1

        for m in self.motors_list:
            m.getPositionSensor().enable(sampling_period)
            m.enableForceFeedback(sampling_period)
            m.enableTorqueFeedback(sampling_period)

    def set_arm_torques(self, torques):
        for i, motor in enumerate(self.motors_list):
            motor.setTorque(torques[i])

    def set_gripper_pos(self, state='on'):
        ''' state : set vacuum gripper "on" or "off" for vacuum activation'''
        if state == 'on' or state == 'On' or state == 'ON':
            self.vac.turnOn()
        else:
            self.vac.turnOff()

    def set_arm_pos(self, pos):
        for i, motor in enumerate(self.motors_list):
            motor.setPosition(pos[i])

    def get_arm_pos(self):
        p = [m.getPositionSensor().getValue() for m in self.motors_list]
        return p

    def get_gripper_pos(self):
        p = [m.getPositionSensor().getValue() for m in self.gripper_list]
        return p

    def get_EE_position(self):
        return self.gps.value


## robot instance
ur5 = UR5e()

a_base = [0, 0, 0, 0, 0, 0]
ur5.set_arm_pos(a_base)
ur5.set_gripper_pos(state='on')

robot.step(TIME_STEP)

img = get_rgb_frame()

final_pos = [-1.3, 2, 1, 2, 1.1, 0]

cnn_model = FeatureExtractorCNN(output_dim=64)

def start_robot(cl, tgs, bs, ms, ks):
    # Import DQN model
    loaded_model = QNetwork()
    loaded_model.load_state_dict(torch.load("dqn_policy_weights.pth"))
    loaded_model.eval()

    # Import YOLO model
    trained_model = YOLO("webot_objects_detection.onnx", task="detect")
    while robot.step(TIME_STEP) != -1:
        ur5.set_arm_pos(final_pos)
        img = get_rgb_frame()
        camera.saveImage("pic.jpeg", 80)

        if (ur5.get_arm_pos()[0] - final_pos[0] < 0.0001) and (ur5.get_arm_pos()[1] - final_pos[1] < 0.0001):

            image = cv2.imread("pic.jpeg")

            # create feature state
            raw_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            transform = transforms.Compose([transforms.ToTensor()])
            input_image = transform(raw_image).unsqueeze(0)
            feature_vector = cnn_model(input_image)

            # use trained Policy Q-Network to apply best action
            with torch.no_grad():
                action_values = loaded_model(feature_vector)
            loaded_model.train()
            action_idx = np.argmax(action_values.cpu().data.numpy())

            # Apply Enhancement
            enhanced_image = image_enhance_action(action_idx, image, cl, tgs, bs, ms, ks)

            # Use YOLO to detect object
            objects = trained_model.predict(source=enhanced_image, save=True, save_txt=True)

            boxes = objects[0].boxes
            objects_bb_data = []

            for box in boxes:
                cls = np.array(box.cls)
                bb = np.array(box.xywhn)
                bb_data = np.append(cls, bb)
                objects_bb_data.append(bb_data)

            print("objects_data :", objects_bb_data)
            X_a, Y_a = -0.150562, -0.21049
            X_b, Y_b = -1.25122, 0.377894

            real_data = []
            for i in range(len(objects_bb_data)):
                x_real = X_a + (X_b - X_a) * (1 - objects_bb_data[i][2])
                y_real = Y_a + (Y_b - Y_a) * (objects_bb_data[i][1])

                real_data.append([objects_bb_data[i][0], x_real, y_real])

            print("\n", "real object data :", real_data)
            break



######################################################  GUI

# Create Window
window = CTk()
window.title("Machine Vision Final Project")

window.grid_rowconfigure(0, weight=1)
window.grid_rowconfigure(1, weight=10)
window.grid_columnconfigure(0, weight=1)

window.minsize(800, 1000)
window.after(0, lambda: window.state("zoomed"))

my_font = CTkFont(family="Segoe UI Black", size=20)

f1 = CTkFrame(window, fg_color="#e4fff3")
f1.grid(row=0, column=0, sticky="nsew")

f2 = CTkFrame(window)
f2.grid(row=1, column=0, sticky="nsew")

txt1 = CTkLabel(f1, text="Machine Vision Project - Dr. Shariatmadar - Jan 2025",
                text_color="black", font=my_font)
txt2 = CTkLabel(f1, text="Hamid Reza Heidari - Milad Mohammadi",
                text_color="black", font=my_font)
txt3 = CTkLabel(f1, text="DQN+CNN Approach for object detection for pick-and-place Operation",
                text_color="black", font=my_font)
txt1.pack(pady=10)
txt2.pack(pady=10)
txt3.pack(pady=10)

f2.grid_columnconfigure(0, weight=1)
f2.grid_columnconfigure(1, weight=3)
f2.grid_rowconfigure(0, weight=1)

f21 = CTkFrame(f2)
f21.grid(row=0, column=0, sticky="nsew")

f22 = CTkFrame(f2)
f22.grid(row=0, column=1, sticky="nsew")

slider_font = CTkFont(family="Dosis ExtraBold", size=22)
mode_font = CTkFont(family="Oswald Medium", size=20)


def update_value_label1(value):
    value_label1.configure(text=f"{int(value)}")  # Display as an integer

frame211 = CTkFrame(f21, height=100)
frame211.pack(pady=10, padx=20, fill="both")

name_label = CTkLabel(frame211, text="(CLAHE) Clip Limit:", anchor="w", font=slider_font)
name_label.grid(row=0, column=0, padx=10, sticky="nsew")

value_label1 = CTkLabel(frame211, text="4", anchor="e", font=slider_font)
value_label1.grid(row=0, column=2, padx=10)

slider1 = CTkSlider(frame211, from_=1, to=10, number_of_steps=10, command=update_value_label1, variable=IntVar(value=4),
                   progress_color="#aaedf1", height=20)
slider1.grid(row=0, column=1, sticky="ew", padx=10)

frame211.grid_columnconfigure(0, weight=1)
frame211.grid_columnconfigure(1, weight=4)
frame211.grid_columnconfigure(2, weight=1)


def update_value_label2(value):
    value_label2.configure(text=f"{float(value / 20):.1f}")  # Display as an integer


frame212 = CTkFrame(f21)
frame212.pack(pady=10, padx=20, fill="x")

name_label = CTkLabel(frame212, text="(CLAHE) Tile Grid Size:", anchor="w", font=slider_font)
name_label.grid(row=0, column=0, padx=10, sticky="nsew")

value_label2 = CTkLabel(frame212, text="4", anchor="e", font=slider_font)
value_label2.grid(row=0, column=2, padx=10)

slider2 = CTkSlider(frame212, from_=1, to=12, number_of_steps=12, command=update_value_label2, variable=IntVar(value=4),
                   progress_color="#aaedf1", height=20)
slider2.grid(row=0, column=1, padx=10, sticky="ew")

frame212.grid_columnconfigure(0, weight=1)
frame212.grid_columnconfigure(1, weight=3)
frame212.grid_columnconfigure(2, weight=1)


def update_value_label3(value):
    value_label3.configure(text=f"{int(value)}")  # Display as an integer


frame213 = CTkFrame(f21)
frame213.pack(pady=10, padx=20, fill="x")

name_label = CTkLabel(frame213, text="(Saturation HSV) Base Scale:", anchor="w", font=slider_font)
name_label.grid(row=0, column=0, padx=10, sticky="nsew")

value_label3 = CTkLabel(frame213, text="1", anchor="e", font=slider_font)
value_label3.grid(row=0, column=2, padx=10)

slider3 = CTkSlider(frame213, from_=1, to=10, number_of_steps=10, command=update_value_label3, variable=IntVar(value=1),
                   progress_color="#aaedf1", height=20)
slider3.grid(row=0, column=1, padx=10, sticky="ew")

frame213.grid_columnconfigure(0, weight=1)
frame213.grid_columnconfigure(1, weight=3)
frame213.grid_columnconfigure(2, weight=1)


def update_value_label4(value):
    value_label4.configure(text=f"{int(value)}")  # Display as an integer


frame214 = CTkFrame(f21)
frame214.pack(pady=10, padx=20, fill="x")

name_label = CTkLabel(frame214, text="(Saturation HSV) Max Scale:", anchor="w", font=slider_font)
name_label.grid(row=0, column=0, padx=10, sticky="nsew")

value_label4 = CTkLabel(frame214, text="8", anchor="e", font=slider_font)
value_label4.grid(row=0, column=2, padx=10)

slider4 = CTkSlider(frame214, from_=2, to=16, number_of_steps=14, command=update_value_label4,
                   variable=IntVar(value=8), progress_color="#aaedf1", height=20)
slider4.grid(row=0, column=1, padx=10, sticky="ew")

frame214.grid_columnconfigure(0, weight=1)
frame214.grid_columnconfigure(1, weight=3)
frame214.grid_columnconfigure(2, weight=1)


def update_value_label5(value):
    value_label5.configure(text=f"{int(value)}")  # Display as an integer


frame215 = CTkFrame(f21)
frame215.pack(pady=10, padx=20, fill="x")

name_label = CTkLabel(frame215, text="(Denoise Gaussian) Kernel Size:", anchor="w", font=slider_font)
name_label.grid(row=0, column=0, padx=10, sticky="nsew")

value_label5 = CTkLabel(frame215, text="9", anchor="e", font=slider_font)
value_label5.grid(row=0, column=2, padx=10)

slider5 = CTkSlider(frame215, height=20, from_=3, to=15, number_of_steps=6, command=update_value_label5,
                   variable=IntVar(value=9), progress_color="#aaedf1")
slider5.grid(row=0, column=1, padx=5, sticky="ew")

frame215.grid_columnconfigure(0, weight=1)
frame215.grid_columnconfigure(1, weight=3)
frame215.grid_columnconfigure(2, weight=1)

def start():

    for widget in f22.winfo_children():
        widget.destroy()

    bs = int(value_label3.cget("text"))
    ms = int(value_label4.cget("text"))
    ks = int(value_label5.cget("text"))
    tgs = int(value_label2.cget("text"))
    cl = int(value_label1.cget("text"))

    start_robot(cl, tgs, bs, ms, ks)

    image = Image.open("./runs/detect/predict/image0.jpg")

    pic_label = CTkLabel(f22, text="")
    pic_label.pack(pady=10)

    photo = CTkImage(image, size=(400, 400))
    pic_label.configure(image=photo, anchor="center")

button2 = CTkButton(f21, text="Start", command=start, fg_color="red", height=35, width=150,
                    corner_radius=10, font=slider_font)
button2.pack(pady=10)

pic_label = CTkLabel(f22, text="")
pic_label.pack(pady=10)
window.mainloop()

