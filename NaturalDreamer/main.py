import math
import time

import gymnasium as gym
import torch
import argparse
import os

from torch.backends import cudnn
from tqdm.auto import tqdm  # add
from dreamer    import Dreamer
from utils      import loadConfig, seedEverything, plotMetrics, _now_sync
from envs import getEnvProperties, GymPixelsProcessingWrapper, CleanGymWrapper, make_env
from utils      import saveLossesToCSV, ensureParentFolders
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
cudnn.allow_tf32 = True
print(device)

def run(configFile):
    config = loadConfig(configFile)
    seedEverything(config.seed)

    runName                 = f"{config.environmentName}_{config.runName}"
    checkpointToLoad        = os.path.join(config.folderNames.checkpointsFolder, f"{runName}_{config.checkpointToLoad}")
    metricsFilename         = os.path.join(config.folderNames.metricsFolder,        runName)
    plotFilename            = os.path.join(config.folderNames.plotsFolder,          runName)
    checkpointFilenameBase  = os.path.join(config.folderNames.checkpointsFolder,    runName)
    videoFilenameBase       = os.path.join(config.folderNames.videosFolder,         runName)
    ensureParentFolders(metricsFilename, plotFilename, checkpointFilenameBase, videoFilenameBase)

    env, envEvaluation = make_env(config)

    observationShape, disc_segments, (cont_dim, actionLow, actionHigh) = getEnvProperties(env) #TODO not disc_n is returned
    print(f"envProperties: obs {observationShape}, cont {cont_dim}, disc {disc_segments}")

    dreamer = Dreamer(observationShape, disc_segments, (cont_dim, actionLow, actionHigh), device, config.dreamer)
    if config.resume:
        dreamer.loadCheckpoint(checkpointToLoad)

    dreamer.environmentInteraction(env, config.episodesBeforeStart, seed=config.seed)

    iteration = 0
    while dreamer.totalGradientSteps <= config.gradientSteps:
        iteration += 1
        mostRecentScore, steps_taken = dreamer.environmentInteraction(env, config.numInteractionEpisodes, seed=config.seed)
        dreamer.prepareEnvironment(env, seed=config.seed)
        steps_taken = steps_taken // config.dreamer.actionRepeat
        training_steps = math.ceil(steps_taken / config.envStepsPerTrainingStep)
        pbar = tqdm(total=training_steps, desc=f"Training for run: {iteration}", unit="step", leave=False)

        for step_idx in range(training_steps):
            sampledData = dreamer.buffer.sample(dreamer.config.batchSize, dreamer.config.batchLength)
            initialStates, worldModelMetrics = dreamer.worldModelTraining(sampledData, enable3dLoss = False)
            behaviorMetrics = dreamer.behaviorTraining(initialStates)

            # show timings on the bar
            dreamer.totalGradientSteps += 1
            pbar.update(1)

            if dreamer.totalGradientSteps % config.checkpointInterval == 0 and config.saveCheckpoints:
                suffix = f"{dreamer.totalGradientSteps/1000:.0f}k"
                dreamer.saveCheckpoint(f"{checkpointFilenameBase}_{suffix}")
                evaluationScore = dreamer.environmentInteraction(envEvaluation, config.numEvaluationEpisodes, seed=config.seed, evaluation=True, saveVideo=True, filename=f"{videoFilenameBase}_{suffix}")
                print(f"Saved Checkpoint and Video at {suffix:>6} gradient steps.")

        if config.saveMetrics:
            metricsBase = {"envSteps": dreamer.totalEnvSteps, "gradientSteps": dreamer.totalGradientSteps, "totalReward" : mostRecentScore}
            saveLossesToCSV(metricsFilename, metricsBase | worldModelMetrics | behaviorMetrics)
            plotMetrics(f"{metricsFilename}", savePath=f"{plotFilename}", title=f"{config.environmentName}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    # parser.add_argument("--config", type=str, default="car-racing-v3.yml")
    parser.add_argument("--config", type=str, default="minecraft-colab.yml")
    run(parser.parse_args(argv).config)


if __name__ == "__main__":
    main()
