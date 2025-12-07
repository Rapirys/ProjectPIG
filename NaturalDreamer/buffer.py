import attridict
import numpy as np
import torch

from malmoenv.world_tracking.world_buffer import WorldBuffer


class ReplayBuffer(object):
    def __init__(self, observation_shape, actions_size, config, device):
        self.config = config
        self.device = device
        self.capacity = int(self.config.capacity)
        self.scene_size = config

        self.observations        = np.empty((self.capacity, *observation_shape), dtype=np.float32)
        self.actions             = np.empty((self.capacity, actions_size), dtype=np.float32)
        self.rewards             = np.empty((self.capacity, 1), dtype=np.float32)
        self.is_first            = np.empty((self.capacity, 1), dtype=np.float32) #TODO Make is_first - boolean
        self.dones               = np.empty((self.capacity, 1), dtype=np.float32)
        self.world_buffer = WorldBuffer(self.capacity, config.minecraft.scene_size)
        self.camera_position = np.empty((self.capacity, 5), dtype=np.float32) #

        self.bufferIndex = 0
        self.full = False

        self.global_tick = -1
        self.tick_index = np.empty((self.capacity,), dtype=np.int64)

        
    def __len__(self):
        return self.capacity if self.full else self.bufferIndex

    def add(self, observation, world_state, camera_position, action, reward, done, is_first):
        self.observations[self.bufferIndex]     = observation
        self.actions[self.bufferIndex]          = action
        self.rewards[self.bufferIndex]          = reward
        self.dones[self.bufferIndex]            = done
        self.is_first[self.bufferIndex]         = is_first
        self.global_tick += 1
        self.tick_index[self.bufferIndex] = self.global_tick
        self.world_buffer.append_update(world_state, camera_position, is_first, done)

        self.bufferIndex = (self.bufferIndex + 1) % self.capacity
        self.full = self.full or self.bufferIndex == 0

    def sample(self, batchSize, sequenceSize):
        lastFilledIndex = self.bufferIndex - sequenceSize + 1
        assert self.full or (lastFilledIndex > batchSize), "not enough data in the buffer to sample"

        sampleIndex = np.random.randint(0, self.capacity if self.full else lastFilledIndex, batchSize).reshape(-1, 1)
        sequenceLength = np.arange(sequenceSize).reshape(1, -1)
        sampleIndex = (sampleIndex + sequenceLength) % self.capacity

        observations         = torch.as_tensor(self.observations[sampleIndex], device=self.device).float()
        actions  = torch.as_tensor(self.actions[sampleIndex], device=self.device)
        rewards  = torch.as_tensor(self.rewards[sampleIndex], device=self.device)
        dones    = torch.as_tensor(self.dones[sampleIndex], device=self.device)
        first    = torch.as_tensor(self.is_first[sampleIndex], device=self.device)

        world_sample_index = self.tick_index[sampleIndex]
        world_trajectories = self.world_buffer.get_trajectories(world_sample_index, self.is_first[sampleIndex].squeeze(-1).astype(bool))

        return attridict({
                "observations": observations,
                "actions": actions,
                "rewards": rewards,
                "world_trajectories": world_trajectories,
                "dones": dones,
                "is_first": first,
            })


