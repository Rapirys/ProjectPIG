import cv2
import torch
import torch.nn as nn
from torch import profiler
from torch.distributions import kl_divergence, Independent, OneHotCategoricalStraightThrough, Normal
import numpy as np
import os

from utils import symlog
from minecraft.loss import CE_ssc_loss
from minecraft.minecraft import MinecraftSegmentationHead, get_classes
from minecraft.utils import (
    extract_camera_position_from_info,
    build_globalid_to_blockid_lut,
)
from malmoenv.world_tracking.utils import decode_world_update
from networks import RecurrentModel, PriorNet, PosteriorNet, RewardModel, ContinueModel, EncoderConv, DecoderConv, Actor, Critic, \
    HybridActor
from utils import computeLambdaValues, Moments, collapse_world_updates
from buffer import ReplayBuffer
import imageio


class Dreamer:
    # TODO rename disc_n, cont_dim
    def __init__(self, observationShape, disc_segments, continuous, device, config):
        self.observationShape   = observationShape
        cont_dim, actionLow, actionHigh = continuous

        self.disc_segments = list(disc_segments)
        self.actionContDim = cont_dim
        self.actionDiscDim = sum(self.disc_segments)
        self.actionSize    = self.actionContDim + self.actionDiscDim

        self.config             = config
        self.device             = device

        self.recurrentSize  = config.recurrentSize
        self.latentSize     = config.latentLength*config.latentClasses
        self.fullStateSize  = config.recurrentSize + self.latentSize

        self.actor = HybridActor(self.fullStateSize, cont_dim, actionLow, actionHigh,
                                 self.disc_segments, device, config.actor).to(self.device)
        self.critic          = Critic(self.fullStateSize,                                                                            config.critic         ).to(self.device)
        self.encoder         = EncoderConv(observationShape, self.config.encodedObsSize,                                             config.encoder        ).to(self.device)
        self.decoder         = DecoderConv(self.fullStateSize, observationShape,                                                     config.decoder        ).to(self.device)
        self.recurrentModel  = RecurrentModel(config.recurrentSize, self.latentSize, self.actionSize,                                     config.recurrentModel ).to(self.device)
        self.priorNet        = PriorNet(config.recurrentSize, config.latentLength, config.latentClasses,                             config.priorNet       ).to(self.device)
        self.posteriorNet    = PosteriorNet(config.recurrentSize + config.encodedObsSize, config.latentLength, config.latentClasses, config.posteriorNet   ).to(self.device)
        self.rewardPredictor = RewardModel(self.fullStateSize,                                                                       config.reward         ).to(self.device)

        self.buffer         = ReplayBuffer(observationShape, self.actionSize, config.buffer, device)
        self.valueMoments   = Moments(device)


        self.block_state_registry = None
        self.block_state_registry_lut = None
        self.classes = config.minecraft.classes
        self.minecraftSegmentationHead = MinecraftSegmentationHead(self.config, self.device, self.fullStateSize,
                                                                   self.observationShape, self.classes).to(self.device)

        self.worldModelParameters = (list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.recurrentModel.parameters()) +
                                     list(self.priorNet.parameters()) + list(self.posteriorNet.parameters()) + list(self.rewardPredictor.parameters()))


        self.worldModelOptimizer    = torch.optim.Adam(self.worldModelParameters,   lr=self.config.worldModelLR)
        self.minecraftHeadOptimiser = torch.optim.Adam(self.minecraftSegmentationHead.parameters(), self.config.worldModelLR)
        self.actorOptimizer         = torch.optim.Adam(self.actor.parameters(),     lr=self.config.actorLR)
        self.criticOptimizer        = torch.optim.Adam(self.critic.parameters(),    lr=self.config.criticLR)

        # self.amp_enabled = device.type == "cuda"
        # self.amp_dtype = (torch.bfloat16 if (self.amp_enabled and torch.cuda.is_bf16_supported()) else torch.float16)
        # self.amp_scaler = torch.amp.GradScaler("cude", enabled=self.amp_enabled)

        self.totalEpisodes      = 0
        self.totalEnvSteps      = 0
        self.totalGradientSteps = 0


    def _flat_action(self, a: dict):
        # TODO Review if there a better place
        parts = []
        if "control" in a: parts.append(a["control"])
        if "macro"   in a: parts.append(a["macro"])
        return torch.cat(parts, -1) if parts else None

    def worldModelTraining(self, data, enable3dLoss = True):
        encodedObservations = self.encoder(data.observations.view(-1, *self.observationShape)).view(self.config.batchSize, self.config.batchLength, -1)
        previousRecurrentState = torch.zeros(self.config.batchSize, self.recurrentSize, device=self.device)
        previousLatentState = torch.zeros(self.config.batchSize, self.latentSize, device=self.device)

        recurrentStates, priorsLogits, posteriors, posteriorsLogits = [], [], [], []
        reconstruction3DLoss = 0
        reconstruction3D_CE = 0.0
        for t in range(1, self.config.batchLength):
            first_t = data.is_first[:, t].float()

            # Reset the recurrent + latent state for rows that start here
            previousRecurrentState = previousRecurrentState * (1.0 - first_t)
            previousLatentState = previousLatentState * (1.0 - first_t)

            recurrentState              = self.recurrentModel(previousRecurrentState, previousLatentState, data.actions[:, t-1])
            _, priorLogits              = self.priorNet(recurrentState)
            posterior, posteriorLogits  = self.posteriorNet(torch.cat((recurrentState, encodedObservations[:, t]), -1))

            recurrentStates.append(recurrentState)
            priorsLogits.append(priorLogits)
            posteriors.append(posterior)
            posteriorsLogits.append(posteriorLogits)

            previousRecurrentState = recurrentState
            previousLatentState = posterior

            camera_position, grid_origin, grid = next(data.world_trajectories)
            camera_position = torch.from_numpy(camera_position).to(self.device)
            grid_origin = torch.from_numpy(grid_origin).to(self.device)
            grid = torch.from_numpy(grid).to(self.device)  # torch.Size([1, 20, 20, 20])
            block_id_grid = self.block_state_registry_lut[grid.long()] #TODO Do not need this mapping in final version
            full_state_step = torch.cat((recurrentState, posterior), dim=-1)

            ssc_input = full_state_step.detach().requires_grad_(True)
            if enable3dLoss:
                reconstruction3DLatent, mask3d = self.minecraftSegmentationHead(ssc_input, camera_position, grid_origin)

                num_classes = reconstruction3DLatent.shape[1]
                class_weight = torch.ones(num_classes, device=self.device)
                #TODO may be croped on top
                target = block_id_grid.masked_fill_(~mask3d.reshape(block_id_grid.shape), -1)
                loss3d = CE_ssc_loss(reconstruction3DLatent, target, class_weight)
                reconstruction3D_CE += loss3d.detach()

                self.minecraftHeadOptimiser.zero_grad(set_to_none=True)
                loss3d.backward()
                ssc_gradients = ssc_input.grad.detach()
                self.minecraftHeadOptimiser.step()
                reconstruction3DLoss = reconstruction3DLoss + (full_state_step * ssc_gradients).sum()

        recurrentStates             = torch.stack(recurrentStates,              dim=1) # (batchSize, batchLength-1, recurrentSize)
        priorsLogits                = torch.stack(priorsLogits,                 dim=1) # (batchSize, batchLength-1, latentLength, latentClasses)
        posteriors                  = torch.stack(posteriors,                   dim=1) # (batchSize, batchLength-1, latentLength*latentClasses)
        posteriorsLogits            = torch.stack(posteriorsLogits,             dim=1) # (batchSize, batchLength-1, latentLength, latentClasses)
        fullStates                  = torch.cat((recurrentStates, posteriors), dim=-1) # (batchSize, batchLength-1, recurrentSize + latentLength*latentClasses)


        reconstructionMeans = self.decoder(fullStates.view(-1, self.fullStateSize)).view(self.config.batchSize, self.config.batchLength-1, *self.observationShape)
        reconstructionMeans = symlog(reconstructionMeans)
        reconstructionLoss = -Normal(reconstructionMeans, 1.0).log_prob(symlog(data.observations[:, 1:])).mean()

        rewardDistribution  =  self.rewardPredictor(fullStates)
        rewardLoss          = -rewardDistribution.log_prob(data.rewards[:, 1:].squeeze(-1)).mean()

        priorDistribution       = Independent(OneHotCategoricalStraightThrough(logits=priorsLogits              ), 1)
        priorDistributionSG     = Independent(OneHotCategoricalStraightThrough(logits=priorsLogits.detach()     ), 1)
        posteriorDistribution   = Independent(OneHotCategoricalStraightThrough(logits=posteriorsLogits          ), 1)
        posteriorDistributionSG = Independent(OneHotCategoricalStraightThrough(logits=posteriorsLogits.detach() ), 1)

        priorLoss       = kl_divergence(posteriorDistributionSG, priorDistribution  )
        posteriorLoss   = kl_divergence(posteriorDistribution  , priorDistributionSG)
        freeNats        = torch.full_like(priorLoss, self.config.freeNats)

        priorLoss       = self.config.betaPrior*torch.maximum(priorLoss, freeNats)
        posteriorLoss   = self.config.betaPosterior*torch.maximum(posteriorLoss, freeNats)
        klLoss          = (priorLoss + posteriorLoss).mean()
        reconstruction3DLoss = reconstruction3DLoss / (self.config.batchLength - 1)
        reconstruction3D_CE = reconstruction3D_CE / (self.config.batchLength - 1)

        worldModelLoss =  reconstructionLoss + rewardLoss + klLoss + reconstruction3DLoss # I think that the reconstruction loss is relatively a bit too high (11k)

        self.worldModelOptimizer.zero_grad()
        #TODO Add loss masking
        worldModelLoss.backward()
        nn.utils.clip_grad_norm_(self.worldModelParameters, self.config.gradientClip, norm_type=self.config.gradientNormType)
        self.worldModelOptimizer.step()

        klLossShiftForGraphing = (self.config.betaPrior + self.config.betaPosterior)*self.config.freeNats
        metrics = {
            "worldModelLoss"        : worldModelLoss.item() - klLossShiftForGraphing,
            "reconstructionLoss"    : reconstructionLoss.item(),
            "reconstruction3DLoss"  :  reconstruction3D_CE.item() if enable3dLoss else 0,
            "rewardPredictorLoss"   : rewardLoss.item(),
            "klLoss"                : klLoss.item() - klLossShiftForGraphing}
        return fullStates.view(-1, self.fullStateSize).detach(), metrics


    def behaviorTraining(self, fullState):
        recurrentState, latentState = torch.split(fullState, (self.recurrentSize, self.latentSize), -1)
        fullStates, logprobs, entropies = [], [], []
        for _ in range(self.config.imaginationHorizon):
            actionFlat, logprob, entropy = self.actor(fullState.detach(), training=True)
            recurrentState = self.recurrentModel(recurrentState, latentState, actionFlat)
            latentState, _ = self.priorNet(recurrentState)

            fullState = torch.cat((recurrentState, latentState), -1)
            fullStates.append(fullState)
            logprobs.append(logprob)
            entropies.append(entropy)
        fullStates  = torch.stack(fullStates,    dim=1) # (batchSize*batchLength, imaginationHorizon, recurrentSize + latentLength*latentClasses)
        logprobs    = torch.stack(logprobs[1:],  dim=1) # (batchSize*batchLength, imaginationHorizon-1)
        entropies   = torch.stack(entropies[1:], dim=1) # (batchSize*batchLength, imaginationHorizon-1)
        
        predictedRewards = self.rewardPredictor(fullStates[:, :-1]).mean
        values           = self.critic(fullStates).mean
        continues        = torch.full_like(predictedRewards, self.config.discount)
        lambdaValues     = computeLambdaValues(predictedRewards, values, continues, self.config.lambda_)

        _, inverseScale = self.valueMoments(lambdaValues)
        advantages      = (lambdaValues - values[:, :-1])/inverseScale

        actorLoss = -torch.mean(advantages.detach()*logprobs + self.config.entropyScale*entropies)

        self.actorOptimizer.zero_grad()
        actorLoss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.gradientClip, norm_type=self.config.gradientNormType)
        self.actorOptimizer.step()

        valueDistributions  =  self.critic(fullStates[:, :-1].detach())
        criticLoss          = -torch.mean(valueDistributions.log_prob(lambdaValues.detach()))

        self.criticOptimizer.zero_grad()
        criticLoss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.gradientClip, norm_type=self.config.gradientNormType)
        self.criticOptimizer.step()

        metrics = {
            "actorLoss"     : actorLoss.item(),
            "criticLoss"    : criticLoss.item(),
            "entropies"     : entropies.mean().item(),
            "logprobs"      : logprobs.mean().item(),
            "advantages"    : advantages.mean().item(),
            "criticValues"  : values.mean().item()}
        return metrics


    @torch.no_grad()
    def environmentInteraction(self, env, numEpisodes,
                               seed=None, evaluation=False, saveVideo=False, filename="videos/unnamedVideo", fps=30, macroBlockSize=16):
        action_repeat = self.config.actionRepeat
        scores = []
        for i in range(numEpisodes):
            recurrentState = torch.zeros(1, self.recurrentSize, device=self.device)
            latentState    = torch.zeros(1, self.latentSize,    device=self.device)
            prevActionVec  = torch.zeros(1, self.actionSize,    device=self.device)

            observation, info = env.reset(seed= (seed + self.totalEpisodes if seed else None))

            registry = info["BlockStateRegistry"]
            if not self.block_state_registry:
                assert len(get_classes(registry)) == self.classes
                self.block_state_registry = registry
                self.block_state_registry_lut = build_globalid_to_blockid_lut(self.block_state_registry, self.device)


            world_state = decode_world_update(info["world_observation"])
            camera_position = extract_camera_position_from_info(info)
            encodedObservation = self.encoder(torch.from_numpy(observation).float().unsqueeze(0).to(self.device))

            currentScore, stepCount, done, is_first, frames = 0, 0, False, True, []
            while not done:
                recurrentState = self.recurrentModel(recurrentState, latentState, prevActionVec)
                latentState, _ = self.posteriorNet(torch.cat((recurrentState, encodedObservation.view(1, -1)), -1))

                actionFlat, _, _ = self.actor(torch.cat((recurrentState, latentState), -1))
                prevActionVec = actionFlat

                totalReward = 0.0
                world_updates = []

                for rep in range(action_repeat):
                    nextObservation, reward, done, info = env.step(actionFlat.cpu().numpy().reshape(-1))
                    totalReward += reward

                    # stream update per low-level tick
                    next_world_state = decode_world_update(info["world_observation"])
                    next_camera_position = extract_camera_position_from_info(info)
                    world_updates.append(next_world_state)

                    if saveVideo and i == 0:
                        frame = env.render()
                        targetHeight = (frame.shape[0] + macroBlockSize - 1)//macroBlockSize*macroBlockSize
                        targetWidth = (frame.shape[1] + macroBlockSize - 1)//macroBlockSize*macroBlockSize
                        frames.append(np.pad(frame, ((0, targetHeight - frame.shape[0]), (0, targetWidth - frame.shape[1]), (0, 0)), mode='edge'))

                    stepCount += 1
                    if done:
                        break


                if not evaluation:
                    self.buffer.add(observation, world_state, camera_position, prevActionVec.cpu().numpy().reshape(-1), totalReward, done, is_first)
                    world_state = collapse_world_updates(world_updates)
                    camera_position = next_camera_position
                is_first = False

                encodedObservation = self.encoder(torch.from_numpy(nextObservation).float().unsqueeze(0).to(self.device))
                observation = nextObservation
                currentScore += totalReward

                if done:
                    scores.append(currentScore)
                    if not evaluation:
                        self.totalEpisodes += 1
                        self.totalEnvSteps += stepCount

                    if saveVideo and i == 0:
                        finalFilename = f"{filename}_reward_{currentScore:.0f}.mp4"
                        with imageio.get_writer(finalFilename, fps=fps) as video:
                            for frame in frames:
                                video.append_data(frame)
                    break
        return sum(scores)/numEpisodes if numEpisodes else None, stepCount

    def saveCheckpoint(self, checkpointPath):
        if not checkpointPath.endswith('.pth'):
            checkpointPath += '.pth'

        checkpoint = {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "recurrentModel": self.recurrentModel.state_dict(),
            "priorNet": self.priorNet.state_dict(),
            "posteriorNet": self.posteriorNet.state_dict(),
            "rewardPredictor": self.rewardPredictor.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "minecraftSegmentationHead": self.minecraftSegmentationHead.state_dict(),
            "minecraftHeadOptimiser": self.minecraftHeadOptimiser.state_dict(),
            "worldModelOptimizer": self.worldModelOptimizer.state_dict(),
            "criticOptimizer": self.criticOptimizer.state_dict(),
            "actorOptimizer": self.actorOptimizer.state_dict(),
            "totalEpisodes": self.totalEpisodes,
            "totalEnvSteps": self.totalEnvSteps,
            "totalGradientSteps": self.totalGradientSteps,
        }
        torch.save(checkpoint, checkpointPath)

    def loadCheckpoint(self, checkpointPath):
        if not checkpointPath.endswith(".pth"):
            checkpointPath += ".pth"
        if not os.path.exists(checkpointPath):
            raise FileNotFoundError(f"Checkpoint file not found at: {checkpointPath}")


        checkpoint = torch.load(checkpointPath, map_location='cpu')
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.decoder.load_state_dict(checkpoint["decoder"])
        self.recurrentModel.load_state_dict(checkpoint["recurrentModel"])
        self.priorNet.load_state_dict(checkpoint["priorNet"])
        self.posteriorNet.load_state_dict(checkpoint["posteriorNet"])
        self.rewardPredictor.load_state_dict(checkpoint["rewardPredictor"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.minecraftSegmentationHead.load_state_dict(checkpoint["minecraftSegmentationHead"])
        self.minecraftHeadOptimiser.load_state_dict(checkpoint["minecraftHeadOptimiser"])
        self.worldModelOptimizer.load_state_dict(checkpoint["worldModelOptimizer"])
        self.criticOptimizer.load_state_dict(checkpoint["criticOptimizer"])
        self.actorOptimizer.load_state_dict(checkpoint["actorOptimizer"])
        self.totalEpisodes = checkpoint["totalEpisodes"]
        self.totalEnvSteps = checkpoint["totalEnvSteps"]
        self.totalGradientSteps = checkpoint["totalGradientSteps"]
