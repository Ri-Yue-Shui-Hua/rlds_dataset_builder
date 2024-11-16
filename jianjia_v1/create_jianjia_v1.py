import numpy as np
import tqdm
import os
import json
from PIL import Image
    

def create_my_episode(file_folder, path):
    json_file = os.path.join(file_folder, "robot_motion_data.json")
    with open(json_file, "r") as f:
        info = json.load(f)
        f.close()
    EPISODE_LENGTH = len(info['steps'])
    episode = []
    for step in range(EPISODE_LENGTH):
        step_info = info['steps'][f'{step}']
        # 1. image
        image_filename = step_info['observation']['image_filename']
        image_path = os.path.join(file_folder, image_filename)
        image = np.array(Image.open(image_path))
        # 2. language instruction
        language_instruction = step_info['natural_language_instruction']
        # 3. action
        delta_position = step_info['action']['delta_x_wrt_base']['delta_position']
        delta_orientation_rpy = step_info['action']['delta_x_wrt_base']['delta_orientation_rpy']
        delta_gripper_closed = step_info['action']['delta_gripper_closed']
        action = delta_position + delta_orientation_rpy
        action.append(delta_gripper_closed)
        # 4. state
        joint_positions = step_info['observation']['joint_positions']
        position = step_info['observation']['end_effector_pose']['position']
        orientation_rpy = step_info['observation']['end_effector_pose']['orientation_rpy']
        gripper_status = step_info['observation']['gripper_status']
        state = joint_positions + position + orientation_rpy
        state.append(gripper_status)   
        
        
        episode.append({
            'image': np.asarray(image, dtype=np.uint8),
            # 'wrist_image': np.asarray(np.random.rand(64, 64, 3) * 255, dtype=np.uint8),
            'state': np.asarray(state, dtype=np.float32),
            'action': np.asarray(action, dtype=np.float32),
            'language_instruction': language_instruction,
        })
    np.save(path, episode)


if __name__ == "__main__":
    data_folder = r"E:\Data\OctoData\finetuneData"
    sub_folders = os.listdir(data_folder)
    all_nums = len(sub_folders)
    N_TRAIN_EPISODES = int(all_nums*0.9)
    N_VAL_EPISODES = all_nums - N_TRAIN_EPISODES
    # create fake episodes for train and validation
    print("Generating train examples...")
    os.makedirs('data/train', exist_ok=True)
    for i in tqdm.tqdm(range(N_TRAIN_EPISODES)):
        current_folder = os.path.join(data_folder, sub_folders[i])
        create_my_episode(current_folder, f'data/train/episode_{i}.npy')
    print("Generating val examples...")
    os.makedirs('data/val', exist_ok=True)
    for i in tqdm.tqdm(range(N_VAL_EPISODES)):
        current_folder = os.path.join(data_folder, sub_folders[N_TRAIN_EPISODES+i])
        create_my_episode(current_folder, f'data/val/episode_{i}.npy')
    print('Successfully created example data!')

