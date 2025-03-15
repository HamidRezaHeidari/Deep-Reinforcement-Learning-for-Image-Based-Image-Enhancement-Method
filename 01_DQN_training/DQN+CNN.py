# Machine Vision Final Project - Dr. Shariatmadar - Author: Hamid Reza Heidari / Milad Mohammadi

# import library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque, namedtuple
import random
import time
import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

from ultralytics import YOLO
import cv2

# HyperParameters
capacity = 150
batch_size = 5
gamma = 0.9
learning_rate = 2e-4
sync_rate = 2
optimize_rate = 5
n_state = 64
n_action = 9

alpha = 4
beta = 1

image_per_episode = 50

threshold = 0.75
zeta = 10

crop_percentage = 0.2

Box_confidence = []
BiscuitBox_confidence = []
CAN_confidence = []
Mouse_confidence = []
Phone_confidence = []

# import trained YOLOv8 OD Model (webot_objects_detection.onnx file should be in directory)
trained_model = YOLO("webot_objects_detection.onnx", task="detect")

def object_detection_rating(source):

    objects = trained_model.predict(source=source, verbose=False)

    # labels => {"Biscuit Box": 0, "Box": 1, "Can": 2, "Mouse": 3, "Phone": 4}
    reward = 0
    boxes = objects[0].boxes
    conf = dict()

    def add_to_dict(key, value):
        if key in conf:
            conf[key].append(value)
        else:
            conf[key] = [value]

    for box in boxes:
        cls = int(box.cls)
        prob = float(box.conf)
        add_to_dict(cls, prob)

    for key in conf.keys():
        if key == 0:
            if len(conf[0]) == 2:
                conf_bb = conf[0][0] + conf[0][1]
                r = conf_bb - 2 * threshold
                reward = reward + r
            else:
                r = abs(len(conf[0]) - 2)
                reward = reward - r * zeta

        elif key == 1:
            if len(conf[1]) == 2:
                conf_b = conf[1][0] + conf[1][1]
                r = conf_b - 2 * threshold
                reward = reward + r
            else:
                r = abs(len(conf[1]) - 2)
                reward = reward - r * zeta

        elif key == 2:
            if len(conf[2]) == 1:
                conf_c = conf[2]
                r = conf_c[0] - threshold
                reward = reward + r
            else:
                r = abs(len(conf[2]) - 1)
                reward = reward - r * zeta

        elif key == 3:
            if len(conf[3]) == 1:
                conf_m = conf[3]
                r = conf_m[0] - threshold
                reward = reward + r
            else:
                r = abs(len(conf[3]) - 1)
                reward = reward - r * zeta

        elif key == 4:
            if len(conf[4]) == 1:
                conf_p = conf[4]
                r = conf_p[0] - threshold
                reward = reward + r
            else:
                r = abs(len(conf[4]) - 1)
                reward = reward - r * zeta
        else:
            print("Error 101 arise >>>>")

    return reward


def image_enhance_action(idx, image, crop_img, crop_type, RAW):

    if idx == 0 or idx == 1 or idx == 2 or idx == 3:
        crop_reverse = crop_model.crop_reverse(crop_img, image, crop_type, idx, RAW)
        return crop_reverse

    # CLAHE
    elif idx == 4:
        #HP
        clip_limit = 4.2
        tile_grid_size = (4, 4)

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
        base_scale = 1
        max_scale = 8

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
        kernel_size = 9

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

class ImageCrop:

    # Random Crop Image
    def crop(self, image, crop_percentage):

        height, width, _ = image.shape

        crop_type = random.choice(["left", "right", "top", "bottom"])
        if crop_type == "left":
            crop_width = int(width * crop_percentage)
            cropped_image = image[:, crop_width:]
        elif crop_type == "right":
            crop_width = int(width * crop_percentage)
            cropped_image = image[:, :width - crop_width]
        elif crop_type == "top":
            crop_height = int(height * crop_percentage)
            cropped_image = image[crop_height:, :]
        elif crop_type == "bottom":
            crop_height = int(height * crop_percentage)
            cropped_image = image[:height - crop_height, :]

        cropped_image = cv2.resize(cropped_image, (640, 640))

        return cropped_image, crop_type

    # Add Padding to Image( for wrong actions)
    def process_image_with_padding(self, image, padding_type):

        padded_image = None
        height, width, _ = image.shape
        padding_size = int(0.2 * max(height, width))

        if padding_type == "left":
            padded_image = cv2.copyMakeBorder(image, 0, 0, padding_size, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        elif padding_type == "right":
            padded_image = cv2.copyMakeBorder(image, 0, 0, 0, padding_size, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        elif padding_type == "top":
            padded_image = cv2.copyMakeBorder(image, padding_size, 0, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        elif padding_type == "bottom":
            padded_image = cv2.copyMakeBorder(image, 0, padding_size, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        resized_padded_image = cv2.resize(padded_image, (640, 640))

        return resized_padded_image

    # Reverse Cropped Image w.r.t given Action
    def crop_reverse(self, crop_image, image, crop_type, action, RAW_state):
        enhance_crop_image = None

        if action == 0:
            if RAW_state and crop_type == "top":
                enhance_crop_image = image
            elif RAW_state and crop_type != "top":
                enhance_crop_image = self.process_image_with_padding(crop_image, "top")
            else:
                enhance_crop_image = self.process_image_with_padding(image, "top")

        elif action == 1:
            if RAW_state and crop_type == "bottom":
                enhance_crop_image = image
            elif RAW_state and crop_type != "bottom":
                enhance_crop_image = self.process_image_with_padding(crop_image, "bottom")
            else:
                enhance_crop_image = self.process_image_with_padding(image, "bottom")

        elif action == 2:
            if RAW_state and crop_type == "left":
                enhance_crop_image = image
            elif RAW_state and crop_type != "left":
                enhance_crop_image = self.process_image_with_padding(crop_image, "left")
            else:
                enhance_crop_image = self.process_image_with_padding(image, "left")

        elif action == 3:
            if RAW_state and crop_type == "right":
                enhance_crop_image = image
            elif RAW_state and crop_type != "right":
                enhance_crop_image = self.process_image_with_padding(crop_image, "right")
            else:
                enhance_crop_image = self.process_image_with_padding(image, "right")
        else:
            print("Error 103 arise >>>>")

        return enhance_crop_image

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

# q-Network
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


# Reply Memory
class ReplayMemory:

    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.buffer.append(e)

    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)

        states = torch.from_numpy(
            np.vstack([e.state.detach().numpy() if isinstance(e.state, torch.Tensor) else e.state
                       for e in experiences if e is not None])
        ).float()

        actions = torch.from_numpy(
            np.vstack([e.action.detach().numpy() if isinstance(e.action, torch.Tensor) else e.action
                       for e in experiences if e is not None])
        ).long()

        rewards = torch.from_numpy(
            np.vstack([e.reward.detach().numpy() if isinstance(e.reward, torch.Tensor) else e.reward
                       for e in experiences if e is not None])
        ).float()

        next_states = torch.from_numpy(
            np.vstack([e.next_state.detach().numpy() if isinstance(e.next_state, torch.Tensor) else e.next_state
                       for e in experiences if e is not None])
        ).float()

        dones = torch.from_numpy(
            np.vstack([e.done.detach().numpy() if isinstance(e.done, torch.Tensor) else e.done
                       for e in experiences if e is not None]).astype(np.uint8)
        ).float()

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

crop_model = ImageCrop()
class Agent():

    def __init__(self):

        # Q-Network
        self.policyNN = QNetwork()
        self.targetNN = QNetwork()
        self.optimizer = optim.Adam(self.policyNN.parameters(), lr=learning_rate)

        # Replay buffer
        self.memory = ReplayMemory(capacity)

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay buffer
        self.memory.add(state, action, reward, next_state, done)

    def optimize_check(self, episode_counter):
        # Learn every sync_rate time steps.
        if (episode_counter % optimize_rate == 0 and len(self.memory) > batch_size):
            experiences = self.memory.sample(batch_size)
            self.optimize(experiences, episode_counter)

    def e_greedy(self, state, eps):

        self.policyNN.eval()
        with torch.no_grad():
            action_values = self.policyNN(state)
        self.policyNN.train()

        # Epsilon-greedy action selection
        if random.random() < eps:
            return random.choice(np.arange(n_action)), "greedy"
        else:
            return np.argmax(action_values.cpu().data.numpy()), "NN"

    def optimize(self, experiences, i):

        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.targetNN(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.policyNN(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network weights every TAU learning steps (so every TAU*sync_rate t_step)
        if i % sync_rate == 0:
            self.policyNN.load_state_dict(self.targetNN.state_dict())

class Machine_Vision_Env():

    def step(self, action, raw_image, crop_img, crop_type, RAW ):

        start = time.time()

        enhanced_image = image_enhance_action(action, raw_image, crop_img, crop_type, RAW)
        ODR = object_detection_rating(enhanced_image)

        stop = time.time()

        reward = alpha/(stop-start) + beta*ODR

        info = True
        next_state = np.zeros((1,n_state))
        truncated = True

        return  next_state, reward, truncated, info

def dqn_cnn(folder_path, episodes=200):

    eps_start = 1.0
    eps_end = 0.02

    start = time.time()
    scores = []
    scores_window = deque(maxlen=10)
    list_eps = []
    eps = eps_start
    episode_counter = 1
    loop_counter = 0
    score = 0

    for _ in range(3):
        for file_name in os.listdir(folder_path):

            loop_counter += 1
            agent.optimize_check(episode_counter+1)

            file_path = os.path.join(folder_path, file_name)
            raw_image = cv2.imread(file_path)

            #Apply Crop to RAW image
            if file_name[:3].upper() == "RAW":
                # Random Crop Image
                True_RAW = True
                cropped_image, crop_type = crop_model.crop(raw_image, crop_percentage)

            else:
                True_RAW = False
                crop_type = None
                cropped_image = raw_image

            #Apply CNN
            raw_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            transform = transforms.Compose([transforms.ToTensor()])
            input_image = transform(raw_image).unsqueeze(0)
            feature_vector = cnn_model(input_image)
            state = feature_vector

            # E-Greedy
            action, choice = agent.e_greedy(state, eps)

            # Check for episode terminate
            if loop_counter % image_per_episode == 0:
                done = 1
                episode_counter += 1
                print("episode #:", episode_counter)
            else:
                done = 0

            next_state, reward, truncated, info = env.step(action, raw_image, cropped_image, crop_type, True_RAW)

            agent.step(state, action, reward, next_state, done)

            score += reward

            if done == 1:

                scores_window.append(score)
                scores.append(score)
                list_eps.append(eps)
                score = 0

                # Dynamic Epsilon
                if episode_counter < episodes * 0.1:
                    eps = eps_start
                elif episodes * 0.1 <= episode_counter < episodes * 0.4:
                    eps = max(0, eps - 1 / episodes)
                elif episodes * 0.4 <= episode_counter < episodes * 0.85:
                    eps = max(eps_end, 0.992 * eps)
                else:
                    eps = 0.0001

    time_elapsed = time.time() - start
    print("Time Elapse: {:.2f} seconds".format(time_elapsed))
    return scores, list_eps

cnn_model = FeatureExtractorCNN(output_dim=64)
agent = Agent()
env = Machine_Vision_Env()
scores, list_eps = dqn_cnn("training_dataset")

torch.save(agent.policyNN.state_dict(), "dqn_policy_weights.pth")

# plot
fig1 = plt.figure()
ax = fig1.add_subplot(111)
p = pd.Series(scores)
ma = p.rolling(25).mean()
plt.plot(p, alpha=0.8)
plt.plot(ma)
plt.ylabel('Score')
plt.xlabel('Episode #')

ax2 = ax.twinx()
ax2.set_ylabel('Epsilon')
ax2.plot(pd.Series(list_eps), color="r")
plt.savefig("result1.jpeg")

fig2 = plt.figure()
ax = fig2.add_subplot(111)
p = pd.Series(scores)
ma = p.rolling(25).mean()
plt.plot(p, alpha=0.8)
plt.plot(ma)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig("result2.jpeg")

plt.show()