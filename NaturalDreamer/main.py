import time

import gymnasium as gym
import torch
import argparse
import os
from tqdm.auto import tqdm  # add
from dreamer    import Dreamer
from utils      import loadConfig, seedEverything, plotMetrics
from envs import getEnvProperties, GymPixelsProcessingWrapper, CleanGymWrapper, make_env
from utils      import saveLossesToCSV, ensureParentFolders
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def _now_sync():
    """Wall-clock time with a CUDA sync so GPU kernels are accounted for."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()

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

    iterationsNum = config.gradientSteps // config.replayRatio
    for iteration in range(iterationsNum):
        mostRecentScore, steps_taken = dreamer.environmentInteraction(env, config.numInteractionEpisodes, seed=config.seed)
        pbar = tqdm(total=steps_taken, desc=f"Training for run: {iteration}", unit="step", leave=False)

        for step_idx in range(steps_taken):
            t0 = _now_sync()
            sampledData = dreamer.buffer.sample(dreamer.config.batchSize, dreamer.config.batchLength)
            t1 = _now_sync()

            initialStates, worldModelMetrics = dreamer.worldModelTraining(sampledData)
            t2 = _now_sync()

            behaviorMetrics = dreamer.behaviorTraining(initialStates)
            t3 = _now_sync()

            # durations
            dt_sample = t1 - t0
            dt_wm     = t2 - t1
            dt_beh    = t3 - t2
            dt_step   = t3 - t0

            # show timings on the bar
            pbar.set_postfix(sample=f"{dt_sample:.2f}s", wm=f"{dt_wm:.2f}s", beh=f"{dt_beh:.2f}s", step=f"{dt_step:.2f}s")

            # (optional) print a line when something is abnormally slow
            if dt_step > 3.0:
                print(f"[slow] iter {iteration} step {step_idx}: sample {dt_sample:.2f}s | world {dt_wm:.2f}s | beh {dt_beh:.2f}s | total {dt_step:.2f}s")

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
    parser.add_argument("--config", type=str, default="minecraft-long.yml")
    run(parser.parse_args(argv).config)


if __name__ == "__main__":
    main()
