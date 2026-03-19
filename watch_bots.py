import os

import cv2
import dm_env
from meltingpot import scenario
import numpy as np


def _sample_random_action(action_spec):
  if isinstance(action_spec, dm_env.specs.DiscreteArray):
    return int(np.random.randint(action_spec.num_values))
  if isinstance(action_spec, dm_env.specs.BoundedArray):
    minimum = int(np.asarray(action_spec.minimum).item())
    maximum = int(np.asarray(action_spec.maximum).item())
    return int(np.random.randint(minimum, maximum + 1))
  raise TypeError(f'Unsupported action spec type: {type(action_spec)}')


def _extract_world_frame(observation):
  if isinstance(observation, (list, tuple)) and observation:
    player_obs = observation[0]
    if 'WORLD.RGB' in player_obs:
      return player_obs['WORLD.RGB']
    if 'RGB' in player_obs:
      return player_obs['RGB']
    return next((value for key, value in player_obs.items() if 'RGB' in key),
                None)

  if isinstance(observation, dict):
    if 'WORLD.RGB' in observation:
      return observation['WORLD.RGB']
    if 'RGB' in observation:
      return observation['RGB']
    return next((value for key, value in observation.items() if 'RGB' in key),
                None)

  return None


def main():
  # 1. LOAD A SCENARIO (Map + Bots)
  # "commons_harvest__open_0" puts you in the game with pre-trained bots
  scenario_name = 'clean_up_0'
  print(f'🤖 Loading Scenario: {scenario_name}...')

  # Build transformed scenario so observations are not restricted to focal-only
  # keys. This makes WORLD.RGB available.
  env = scenario.build(scenario_name, substrate_transform=lambda s: s)

  # 2. SETUP
  output_dir = '/home/lloyd/Documents/DL project CT-MRI/2D UNet/DM/meltingpot/trained_gameplay_videos'
  os.makedirs(output_dir, exist_ok=True)
  output_video = os.path.join(output_dir, f'{scenario_name}.mp4')

  total_steps = 400
  capture_every = 1
  output_size = (512, 512)
  output_fps = 12

  timestep = env.reset()

  # The action_spec here is only for focal agents (the slots you control).
  # Background bots move automatically inside the engine.
  action_spec = env.action_spec()
  num_focal_agents = len(action_spec)
  print(f'   -> Setup complete. You control {num_focal_agents} agent(s).')
  print('   -> The background bots are now playing against you.')

  # 3. RUN THE SIMULATION
  # Run and capture frames directly to video.
  print('🎥 Starting simulation & recording video...')

  writer = None
  frames_written = 0

  try:
    for i in range(total_steps):
      # We just have focal agents do random legal actions while we watch bots.
      actions = [_sample_random_action(spec) for spec in action_spec]
      timestep = env.step(actions)

      if i % capture_every != 0:
        continue

      frame_data = _extract_world_frame(timestep.observation)
      if frame_data is None:
        continue

      frame = frame_data.astype(np.uint8)
      frame = cv2.resize(frame, output_size, interpolation=cv2.INTER_NEAREST)
      frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

      if writer is None:
        writer = cv2.VideoWriter(
            output_video,
            cv2.VideoWriter_fourcc(*'mp4v'),
            output_fps,
            output_size,
        )

      writer.write(frame)
      frames_written += 1

    if writer is not None:
      writer.release()
      writer = None

    print(f'✅ Done! Wrote {frames_written} frames to {output_video}')
  finally:
    if writer is not None:
      writer.release()
    env.close()


if __name__ == '__main__':
  main()
