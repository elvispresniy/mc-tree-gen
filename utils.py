from random import shuffle, randint
import os

import nbtlib

import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Schematic:
    def __init__(self, width, height, length, blocks):
        self.width = nbtlib.tag.Short(width)
        self.height = nbtlib.tag.Short(height)
        self.length = nbtlib.tag.Short(length)

        self.blocks = self.nbt_blocks(blocks)
        self.data = self.nbt_data(width * height * length)

    def nbt_blocks(self, blocks):
        b_arr = nbtlib.tag.ByteArray([nbtlib.tag.Byte(block) for block in blocks])
        return b_arr
    
    def nbt_data(self, dimension):
        b_arr = nbtlib.tag.ByteArray([nbtlib.tag.Byte(0) for _ in range(dimension)])
        return b_arr
    
    def __call__(self, path):
        file = nbtlib.load(path)
        
        file['Width'] = self.width
        file['Height'] = self.height
        file['Length'] = self.length
        file['Blocks'] = self.blocks
        file['Data'] = self.data

        file['WEOffsetX'] = nbtlib.tag.Int(0)
        file['WEOffsetY'] = nbtlib.tag.Int(0)
        file['WEOffsetZ'] = nbtlib.tag.Int(0)
        file['WEOriginX'] = nbtlib.tag.Int(0)
        file['WEOriginY'] = nbtlib.tag.Int(0)
        file['WEOriginZ'] = nbtlib.tag.Int(0)
        file['Materials'] = nbtlib.tag.String('Alpha')

        file.save()

def map_color(schem: torch.Tensor, df: pd.DataFrame):
    '''Maps color channels to block ids tensor'''
    r_channel = np.vectorize(df['r'].to_dict().get)
    g_channel = np.vectorize(df['g'].to_dict().get)
    b_channel = np.vectorize(df['b'].to_dict().get)
    a_channel = np.vectorize(df['a'].to_dict().get)

    r_schem = r_channel(schem)
    g_schem = g_channel(schem)
    b_schem = b_channel(schem)
    a_schem = a_channel(schem)

    result = np.concatenate((r_schem, g_schem, b_schem, a_schem))

    return torch.tensor(result) / 255.0

def find_closest_neighbors(voxel_tensor, points_tensor):
    batch_size, n_channels, h, w, l = voxel_tensor.shape

    # Reshape voxel tensor to (batch_size, n_channels, h*w*l) to handle all voxels together
    reshaped_voxel_tensor = voxel_tensor.view(batch_size, n_channels, -1)
    
    # Initialize an empty tensor to store the closest points
    closest_neighbors = torch.empty((batch_size, h, w, l))

    for b in range(batch_size):
        for i in range(h * w * l):
            voxel = reshaped_voxel_tensor[b, ..., i]
            voxel = voxel.unsqueeze(0)  # Shape (1,)

            # Compute the Euclidean distance from the voxel to each point in points_tensor
            distances = torch.norm(points_tensor - voxel, dim=-1)

            # Find the index of the closest point
            closest_idx = torch.argmin(distances)

            # Store the closest point back in the correct shape
            closest_neighbors[b, i // (w * l), (i // l) % w, i % l] = closest_idx

    return closest_neighbors

def schematic_to_tensor(path, key='Blocks'):
    '''Convert .schematic to 3d tensor'''
    file = nbtlib.load(path)
    metadata = (int(file['Height']), int(file['Length']), int(file['Width']),)
    tensor = torch.tensor(file[key]).reshape(metadata).to(torch.uint8)
    return tensor

def schematic_to_binary_tensor(path):
    '''Convert .schematic to binary 3d tensor'''
    file = nbtlib.load(path)
    metadata = (int(file['Height']), int(file['Length']), int(file['Width']),)
    tensor = torch.tensor(file['Blocks']).reshape(metadata)
    converted_tensor = torch.where(tensor != 0, torch.tensor(1.0), torch.tensor(0.0)).to(torch.float32)
    return converted_tensor

def get_all_paths(directory):
    '''Get all .schematic paths from a directory'''
    all_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            all_paths.append(os.path.join(root, file))
    return all_paths

def pad_to_fixed_shape(tensor, target_shape=32):
    '''Pad 3d tensor to a fixed target shape'''
    pad_x = target_shape - tensor.shape[0]
    pad_y = target_shape - tensor.shape[1]
    pad_z = target_shape - tensor.shape[2]

    padding = (
        pad_z - pad_z//2, pad_z//2,
        pad_y - pad_y//2, pad_y//2, 
        pad_x - pad_x//2, pad_x//2,
    )

    tensor = F.pad(tensor, padding, 'constant')

    return(tensor)

def convert_negatives(tensor):
    '''Convert negative numbers inside tensor into 127 - number'''
    mask = tensor < 0

    positive_counterparts = tensor.abs()

    transformed_tensor = torch.where(mask, 127 + positive_counterparts, tensor)

    return transformed_tensor

def convert_negatives_back(tensor):
    '''Convert numbers > 127 inside tensor to 127 - number'''
    mask = tensor > 127

    transformed_tensor = torch.where(mask, 127 - tensor, tensor)

    return transformed_tensor

def get_loader(directory_path: str, split_p: float = 0.8, shape_limit: int = 32, batch_size: int = 32) -> DataLoader:
    '''DataLoader without labels'''
    all_paths = get_all_paths(directory_path)
    
    houses = list()
    for path in all_paths:
        try:
            tensor = schematic_to_tensor(path)
        except:
            continue
        if all(tensor.shape[i] <= shape_limit for i in range(3)):
            normalized_tensor = pad_to_fixed_shape(tensor, shape_limit).to(torch.int64).to(device)
            normalized_tensor = convert_negatives(normalized_tensor)
            houses.append(normalized_tensor)

    split = int(len(houses) * split_p)
        
    trainloader = DataLoader(houses[:split], batch_size=batch_size, shuffle=True)
    testloader = DataLoader(houses[split:], batch_size=batch_size, shuffle=False)
    return trainloader, testloader, houses

def add_labels(tensors: list[torch.Tensor], label: int):
    return list(zip(tensors, [label for _ in range(len(tensors))]))

def get_loader_binary(labeled_tensors: list[torch.Tensor], split: float = 0.8, batch_size: int = 1):
    '''DataLoader with labels'''
    
    split = int(len(labeled_tensors) * split)

    trainloader = DataLoader(labeled_tensors[:split], batch_size=batch_size, shuffle=True)
    testloader = DataLoader(labeled_tensors[split:], batch_size=batch_size, shuffle=False)

    return trainloader, testloader

def visualize(tensor_data: torch.Tensor):
    '''A visualizing function'''
    shape = tensor_data.shape[-1]
    tensor_data = tensor_data.reshape(shape, shape, shape)

    # Filter out the zero values
    non_zero_mask = tensor_data != 0
    non_zero_values = tensor_data[non_zero_mask]

    # Create a meshgrid of 3D coordinates for each element in the tensor
    x, y, z = torch.meshgrid(torch.arange(tensor_data.shape[0]),
                            torch.arange(tensor_data.shape[1]),
                            torch.arange(tensor_data.shape[2]))

    # Flatten the tensor and meshgrid coordinates for plotting
    x = x[non_zero_mask].numpy()
    y = y[non_zero_mask].numpy()
    z = z[non_zero_mask].numpy()

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set the data points as a scatter plot
    ax.scatter(z, y, x, c=non_zero_values, cmap='viridis', marker='o')

    # Set axis labels
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    # Show the plot
    plt.show()

def get_tensors(
        path: str,
        df: pd.DataFrame,
        filter_values: list[int],
        pad_shape: int = 64,
        limit_shape: int = 64
):
    '''Preprocesses every schematic provided at 'path' into tensor'''
    all_paths = get_all_paths(path)

    tensors = []

    for path in all_paths:
        try:
            tensor = schematic_to_tensor(path)
        except:
            print(f'Error processing: {path}')
            continue

        if all(tensor.shape[i] <= limit_shape for i in range(3)):
            normalized_tensor = pad_to_fixed_shape(tensor, pad_shape)[None, :]
            filtered_tensor = filter_tensor(normalized_tensor, filter_values)
            if (18 in filtered_tensor or 161 in filtered_tensor) and (17 in filtered_tensor or 162 in filtered_tensor):
              colored_tensor = map_color(filtered_tensor, df).to(torch.float32)[:2, ...]
              tensors.append(colored_tensor)

    return tensors

def filter_tensor(data: torch.Tensor, filter_values: list[int]):
    '''Replces all blocks that are not included in filter_values with value that indicates air block'''
    values_tensor = torch.tensor(filter_values)
    mask = (data.unsqueeze(-1) == values_tensor).any(-1)
    filtered_data = torch.where(mask, data, torch.tensor(0))
    return filtered_data

def show_pred(v_mesh, log_thresh: float = 0.15, leaves_thresh: float = 0.15):
    '''Transforms, vizualizes and returns a resulting tree tensor'''
    leaves_mesh = (v_mesh[:, 1, ...] > leaves_thresh).cpu() * 18
    log_mesh = (v_mesh[:, 0, ...] > log_thresh).cpu() * 17

    mask = (v_mesh[:, 1, ...] > v_mesh[:, 0, ...]).cpu()

    mesh_result = torch.where(
        mask,
        leaves_mesh,
        log_mesh
    )

    visualize(mesh_result)

    return mesh_result