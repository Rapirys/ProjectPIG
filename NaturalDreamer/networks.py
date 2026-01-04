import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Bernoulli, Independent, OneHotCategoricalStraightThrough
from torch.distributions.utils import probs_to_logits
from utils import sequentialModel1D


class RecurrentModel(nn.Module):
    def __init__(self, recurrentSize, latentSize, actionSize, config):
        super().__init__()
        self.config = config
        self.activation = getattr(nn, self.config.activation)()

        self.input_proj = nn.Linear(latentSize + actionSize, self.config.hiddenSize)

        self.num_blocks = self.config.numBlocks
        self.block_size = recurrentSize // self.num_blocks  # total = num_blocks × block_size (DreamerV3)
        self.block_cells = nn.ModuleList(
            [nn.GRUCell(self.config.hiddenSize, self.block_size) for _ in range(self.num_blocks)]
        )

    def forward(self, recurrent_state, latent_state, action):
        projected_input = self.activation(self.input_proj(torch.cat((latent_state, action), dim=-1)))
        block_states = recurrent_state.split(self.block_size, dim=-1)

        updated_blocks = []
        for block_idx, gru_cell in enumerate(self.block_cells):
            updated_block = gru_cell(projected_input, block_states[block_idx])
            updated_blocks.append(updated_block)

        updated_state = torch.cat(updated_blocks, dim=-1)
        return updated_state



class PriorNet(nn.Module):
    def __init__(self, inputSize, latentLength, latentClasses, config):
        super().__init__()
        self.config = config
        self.latentLength = latentLength
        self.latentClasses = latentClasses
        self.latentSize = latentLength*latentClasses
        self.network = sequentialModel1D(inputSize, [self.config.hiddenSize]*self.config.numLayers, self.latentSize, self.config.activation)
    
    def forward(self, x):
        rawLogits = self.network(x)

        probabilities = rawLogits.view(-1, self.latentLength, self.latentClasses).softmax(-1)
        uniform = torch.ones_like(probabilities)/self.latentClasses
        finalProbabilities = (1 - self.config.uniformMix)*probabilities + self.config.uniformMix*uniform
        logits = probs_to_logits(finalProbabilities)

        sample = Independent(OneHotCategoricalStraightThrough(logits=logits), 1).rsample()
        return sample.view(-1, self.latentSize), logits
    

class PosteriorNet(nn.Module):
    def __init__(self, inputSize, latentLength, latentClasses, config):
        super().__init__()
        self.config = config
        self.latentLength = latentLength
        self.latentClasses = latentClasses
        self.latentSize = latentLength*latentClasses
        self.network = sequentialModel1D(inputSize, [self.config.hiddenSize]*self.config.numLayers, self.latentSize, self.config.activation)
    
    def forward(self, x):
        rawLogits = self.network(x)

        probabilities = rawLogits.view(-1, self.latentLength, self.latentClasses).softmax(-1)
        uniform = torch.ones_like(probabilities)/self.latentClasses
        finalProbabilities = (1 - self.config.uniformMix)*probabilities + self.config.uniformMix*uniform
        logits = probs_to_logits(finalProbabilities)

        sample = Independent(OneHotCategoricalStraightThrough(logits=logits), 1).rsample()
        return sample.view(-1, self.latentSize), logits


class RewardModel(nn.Module):
    def __init__(self, inputSize, config):
        super().__init__()
        self.config = config
        self.network = sequentialModel1D(inputSize, [self.config.hiddenSize]*self.config.numLayers, 2, self.config.activation)

    def forward(self, x):
        mean, logStd = self.network(x).chunk(2, dim=-1)
        return Normal(mean.squeeze(-1), torch.exp(logStd).squeeze(-1))


class ContinueModel(nn.Module):
    def __init__(self, inputSize, config):
        super().__init__()
        self.config = config
        self.network = sequentialModel1D(inputSize, [self.config.hiddenSize]*self.config.numLayers, 1, self.config.activation)

    def forward(self, x):
        return Bernoulli(logits=self.network(x).squeeze(-1))


class EncoderConv(nn.Module):
    def __init__(self, inputShape, outputSize, config):
        super().__init__()
        self.config = config
        activation = getattr(nn, self.config.activation)()
        channels, height, width = inputShape
        self.outputSize = outputSize

        self.convolutionalNet = nn.Sequential(
            nn.Conv2d(channels,            self.config.depth*1, self.config.kernelSize, self.config.stride, padding=1), activation,
            nn.Conv2d(self.config.depth*1, self.config.depth*2, self.config.kernelSize, self.config.stride, padding=1), activation,
            nn.Conv2d(self.config.depth*2, self.config.depth*4, self.config.kernelSize, self.config.stride, padding=1), activation,
            nn.Conv2d(self.config.depth*4, self.config.depth*8, self.config.kernelSize, self.config.stride, padding=1), activation,
            nn.Flatten(),
            nn.Linear(self.config.depth*8*(height // (self.config.stride ** 4))*(width // (self.config.stride ** 4)), outputSize), activation)

    def forward(self, x):
        return self.convolutionalNet(x).view(-1, self.outputSize)


class DecoderConv(nn.Module):
    def __init__(self, inputSize, outputShape, config):
        super().__init__()
        self.config = config
        self.channels, self.height, self.width = outputShape
        activation = getattr(nn, self.config.activation)()

        h4 = self.height // (self.config.stride ** 4)
        w4 = self.width  // (self.config.stride ** 4)
        start_ch = self.config.depth * 8
        kernel = self.config.kernelSize
        size = self.config.stride

        self.network = nn.Sequential(
            nn.Linear(inputSize, start_ch * h4 * w4),
            nn.Unflatten(1, (start_ch, h4, w4)),
            nn.ConvTranspose2d(start_ch, self.config.depth*4, kernel, size, padding=1), activation,
            nn.ConvTranspose2d(self.config.depth*4,  self.config.depth*2, kernel, size, padding=1), activation,
            nn.ConvTranspose2d(self.config.depth*2,  self.config.depth*1, kernel, size, padding=1), activation,
            nn.ConvTranspose2d(self.config.depth*1,  self.channels, kernel, size, padding=1),
        )

    def forward(self, x):
        return self.network(x)


class DecoderDepth(nn.Module):
    def __init__(self, decoder, depth_head):
        super().__init__()
        self.decoder = decoder
        self.depth_head = depth_head

    def forward(self, x):
        features = self.decoder(x)
        return self.depth_head(features), features


class Decoder3d(nn.Module):
    def __init__(self, decoder, minecraftHead):
        super().__init__()
        self.decoder = decoder
        self.minecraftHead = minecraftHead

    def forward(self, depth_features, x, camera_position, model_view_metrix, projection_metrix, grid_origin):
        features = torch.cat([depth_features, self.decoder(x)], dim=1)
        return self.minecraftHead(features, camera_position, model_view_metrix, projection_metrix, grid_origin)

class Decoder3dSparse(nn.Module):
    def __init__(self, decoder, minecraftHead):
        super().__init__()
        self.decoder = decoder
        self.minecraftHead = minecraftHead

    def forward(self,
        depth_features: torch.Tensor,
        full_state: torch.Tensor, # [B, F]  (recurrent+latent)
        mixture_weights: torch.Tensor,      # [B, K, H, W]
        component_means: torch.Tensor,      # [B, K, H, W]
        component_scales: torch.Tensor,     # [B, K, H, W]
        camera_position: torch.Tensor,      # [B, 3]
        model_view_metrix: torch.Tensor,    # [B, 4, 4]
        projection_metrix: torch.Tensor,
        target_coords: torch.Tensor):
        features = torch.cat([depth_features, self.decoder(full_state)], dim=1)
        return self.minecraftHead(features, mixture_weights, component_means, component_scales, camera_position, model_view_metrix, projection_metrix, target_coords)



class Actor(nn.Module):
    def __init__(self, inputSize, actionSize, actionLow, actionHigh, device, config):
        super().__init__()
        actionSize *= 2
        self.config = config
        self.network = sequentialModel1D(inputSize, [self.config.hiddenSize]*self.config.numLayers, actionSize, self.config.activation)
        self.register_buffer("actionScale", ((torch.tensor(actionHigh, device=device) - torch.tensor(actionLow, device=device)) / 2.0))
        self.register_buffer("actionBias",  ((torch.tensor(actionHigh, device=device) + torch.tensor(actionLow, device=device)) / 2.0))

    def forward(self, x):
        logStdMin, logStdMax = -5, 2
        mean, logStd = self.network(x).chunk(2, dim=-1)
        logStd = logStdMin + (logStdMax - logStdMin)/2*(torch.tanh(logStd) + 1) # (-1, 1) to (min, max)
        std = torch.exp(logStd)

        distribution = Normal(mean, std)
        sample = distribution.sample()
        sampleTanh = torch.tanh(sample)
        action = sampleTanh*self.actionScale + self.actionBias

        logprobs = distribution.log_prob(sample)
        logprobs -= torch.log(self.actionScale*(1 - sampleTanh.pow(2)) + 1e-6)
        entropy = distribution.entropy()
        return action, logprobs.sum(-1), entropy.sum(-1)


class DiscreteActor(nn.Module):
    # TODO review DiscreteActor logick
    def __init__(self, inputSize, segments, device, config):
        super().__init__()
        self.config   = config
        self.segments = list(segments)
        self.total    = sum(self.segments)
        self.network  = sequentialModel1D(inputSize,
                           [self.config.hiddenSize]*self.config.numLayers,
                           self.total, self.config.activation)

    def forward(self, x):
        logits = self.network(x)                           # shape: (B, total)
        parts  = torch.split(logits, self.segments, dim=-1)  # list of (B, n_i)

        samples, logps, ents = [], [], []
        for p in parts:
            base = OneHotCategoricalStraightThrough(logits=p)  # each key
            s = base.rsample()                                  # (B, n_i), one-hot
            samples.append(s)
            logps.append(base.log_prob(s))                      # (B,)
            ents.append(base.entropy())                         # (B,)

        action  = torch.cat(samples, dim=-1)                    # (B, total)
        logprob = torch.stack(logps, dim=-1).sum(dim=-1)        # (B,)
        entropy = torch.stack(ents,  dim=-1).sum(dim=-1)        # (B,)
        return action, logprob, entropy


        
class HybridActor(nn.Module):
    def __init__(self, inputSize, cont_dim, cont_low, cont_high, disc_segments, device, config):
        super().__init__()
        self.cont = Actor(inputSize, cont_dim, cont_low, cont_high, device, config) if cont_dim > 0 else None
        self.disc = DiscreteActor(inputSize, disc_segments, device, config)          if sum(disc_segments) > 0 else None
    #     TODO Avoid setting to NONE

    def forward(self, x, training=False):
        heads = [m for m in (self.cont, self.disc) if m is not None]

        actions, logprobs, entropy = [], 0, 0
        for head in heads:
            a, lp, en = head(x)  #DODO
            actions.append(a)
            logprobs += lp
            entropy += en

        flat_actions = torch.cat(actions, -1)
        return flat_actions, logprobs, entropy



class Critic(nn.Module):
    def __init__(self, inputSize, config):
        super().__init__()
        self.config = config
        self.network = sequentialModel1D(inputSize, [self.config.hiddenSize]*self.config.numLayers, 2, self.config.activation)

    def forward(self, x):
        mean, logStd = self.network(x).chunk(2, dim=-1)
        return Normal(mean.squeeze(-1), torch.exp(logStd).squeeze(-1))
