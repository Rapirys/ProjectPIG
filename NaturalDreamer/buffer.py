import attridict
import numpy as np
import torch

from malmoenv.world_tracking.world_buffer import WorldBuffer, WorldTrajectory
from minecraftscc.sparse.data_preparation import visible_blocks_with_halo_numpy


class ReplayBuffer(object):
    def __init__(self, observation_shape, actions_size, config, device, use_3d_predictions = 'sparse'):
        self.main_config = config
        self.config = config.buffer
        self.device = device
        self.capacity = int(self.config.capacity)
        self.scene_size = config
        self.use_3d_predictions = use_3d_predictions
        self.observation_shape = observation_shape

        self.observations = np.empty((self.capacity, *observation_shape), dtype=np.float32)
        self.depths = np.empty((self.capacity, *observation_shape), dtype=np.float32)
        self.actions = np.empty((self.capacity, actions_size), dtype=np.float32)
        self.rewards = np.empty((self.capacity, 1), dtype=np.float32)
        self.is_first = np.empty((self.capacity, 1), dtype=np.float32)
        self.dones = np.empty((self.capacity, 1), dtype=np.float32)
        self.camera_position = np.empty((self.capacity, 3), dtype=np.float32)  #

        if use_3d_predictions == 'dense':
            self.world_buffer = WorldBuffer(self.capacity, config.minecraft.scene_size)
        if use_3d_predictions == 'sparse':
            self.world_trajectory = WorldTrajectory(eager = True)
            self.sparse_coords = np.empty((self.capacity,), dtype=object)
            self.sparse_direct_mask = np.empty((self.capacity,), dtype=object)

        self.model_view_metrix = np.empty((self.capacity, 4, 4), dtype=np.float32)
        self.projection_metrix = np.empty((self.capacity, 4, 4), dtype=np.float32)

        self.bufferIndex = 0
        self.full = False
        self.global_tick = -1
        self.tick_index = np.empty((self.capacity,), dtype=np.int64)
        self.block_state_registry_lut = None

    def __len__(self):
        return self.capacity if self.full else self.bufferIndex


    def add(self, observation, depth, world_state, camera_position, model_view_metrix, projection_metrix, action, reward, done, is_first):
        self.observations[self.bufferIndex] = observation
        self.depths[self.bufferIndex] = depth
        self.actions[self.bufferIndex] = action
        self.rewards[self.bufferIndex] = reward
        self.dones[self.bufferIndex] = done
        self.is_first[self.bufferIndex] = is_first

        self.global_tick += 1
        self.tick_index[self.bufferIndex] = self.global_tick

        if self.use_3d_predictions == 'dense':
            self.world_buffer.append_update(world_state, camera_position, model_view_metrix, projection_metrix, is_first, done)
        if self.use_3d_predictions == 'sparse':
            if is_first:
                self.world_trajectory = WorldTrajectory(eager = True)
            self.world_trajectory.add_update(world_state, camera_position, model_view_metrix, projection_metrix)

            coords, direct_mask = self._compute_sparse_targets(depth, camera_position, model_view_metrix, projection_metrix)
            self.sparse_coords[self.bufferIndex] = coords
            self.sparse_direct_mask[self.bufferIndex] = direct_mask

        self.bufferIndex = (self.bufferIndex + 1) % self.capacity
        self.full = self.full or self.bufferIndex == 0

    def sample(self, batchSize, sequenceSize):
        lastFilledIndex = self.bufferIndex - sequenceSize + 1
        assert self.full or (lastFilledIndex > batchSize), "not enough data in the buffer to sample"

        sampleIndex = np.random.randint(0, self.capacity if self.full else lastFilledIndex, batchSize).reshape(-1, 1)
        sequenceLength = np.arange(sequenceSize).reshape(1, -1)
        sampleIndex = (sampleIndex + sequenceLength) % self.capacity

        observations = torch.as_tensor(self.observations[sampleIndex], device=self.device).float()
        depths = torch.as_tensor(self.depths[sampleIndex], device=self.device).float()
        actions = torch.as_tensor(self.actions[sampleIndex], device=self.device)
        rewards = torch.as_tensor(self.rewards[sampleIndex], device=self.device)
        dones = torch.as_tensor(self.dones[sampleIndex], device=self.device)
        first = torch.as_tensor(self.is_first[sampleIndex], device=self.device)

        world_sample_index = self.tick_index[sampleIndex]

        world_trajectories = None
        if self.use_3d_predictions == 'dense':
            starts_mask = self.is_first[sampleIndex].squeeze(-1).astype(bool)  # Add restart points
            starts_mask[:, 1:] |= world_sample_index[:, 1:] < world_sample_index[:, :-1]  # TODO avoid sampling that crosses trajectory boundary
            world_trajectories = self.world_buffer.get_trajectories(world_sample_index, starts_mask)

        sparse_targets = None
        if self.use_3d_predictions == 'sparse':
            sparse_targets = {"coords": self.sparse_coords[sampleIndex], "direct_mask": self.sparse_direct_mask[sampleIndex]}

        return attridict({
            "observations": observations,
            "depths": depths,
            "actions": actions,
            "rewards": rewards,
            "world_trajectories": world_trajectories,
            "sparse_targets": sparse_targets,
            "dones": dones,
            "is_first": first,
        })

    def _compute_sparse_targets(self, depth: np.ndarray, camera_position, model_view_metrix: np.ndarray, projection_metrix: np.ndarray):
        coords, direct_mask = visible_blocks_with_halo_numpy(
            depth_camera_z=torch.from_numpy(depth).to(device = self.device),
            camera_position=torch.from_numpy(camera_position).to(device = self.device),
            model_view_metrix = torch.from_numpy(model_view_metrix).to(device = self.device),
            projection_metrix = torch.from_numpy(projection_metrix).to(device = self.device),
            image_width=self.observation_shape[1],
            image_height=self.observation_shape[1],
            halo_size=(5, 5, 5),
        )
        coords = coords[:, 1:4]

        state_ids, valid = self.world_trajectory.lookup_block_states(coords)
        coords = coords[valid]
        direct_mask = direct_mask[valid]
        state_ids = state_ids[valid]
        block_ids = self.block_state_registry_lut[state_ids]
        coords = np.concatenate([coords, block_ids[:, None]], axis=1)

        max_n = self.main_config.minecraft.max_target_blocks_3d
        if coords.shape[0] > max_n:
            idx = np.random.choice(coords.shape[0], max_n, replace=False)
            coords, direct_mask = coords[idx], direct_mask[idx]

        return coords, direct_mask