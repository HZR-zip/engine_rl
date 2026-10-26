# SPDX-License-Identifier: BSD-3-Clause
"""Tensor-only observation assembly shared by the Lab adapter and CPU checks."""
import torch
from .contract import COMMAND_SCALE, FRAME_DIM, HISTORY_STEPS, OBS_CLIP


def proprioception(commands, angular_velocity, gravity, joint_delta, joint_velocity, actions):
    return torch.cat((commands * commands.new_tensor(COMMAND_SCALE), angular_velocity * 0.25,
                      gravity, joint_delta, joint_velocity * 0.05, actions), dim=-1)


def corrupt_observations(frame, heights):
    """Author uniform proprio/height noise and four high + four low height outliers."""
    amplitude = frame.new_tensor([0.0] * 3 + [0.05] * 3 + [0.05] * 3
                                 + [0.01] * 12 + [0.075] * 12 + [0.0] * 12)
    frame = frame + (torch.rand_like(frame) * 2 - 1) * amplitude
    heights = heights + (torch.rand_like(heights) * 2 - 1) * 0.15
    low, high = heights.min(-1, keepdim=True).values, heights.max(-1, keepdim=True).values
    ids = torch.rand_like(heights).argsort(-1)[:, :8]
    values = torch.cat((high + torch.rand_like(ids[:, :4], dtype=heights.dtype) * (high - low),
                        low - torch.rand_like(ids[:, 4:], dtype=heights.dtype) * (high - low)), -1)
    return frame, heights.scatter(1, ids, values)


class ObservationHistory:
    """Newest-first frames; reset environments repeat their first frame ten times."""
    def __init__(self, num_envs, device='cpu'):
        self.frames = torch.zeros(num_envs, HISTORY_STEPS, FRAME_DIM, device=device)
        self.fresh = torch.ones(num_envs, dtype=torch.bool, device=device)

    def reset(self, env_ids=None):
        self.fresh[slice(None) if env_ids is None else env_ids] = True

    def push(self, frame, scaled_velocity, force, heights):
        if not torch.isfinite(heights).all():
            raise ValueError('Non-finite height ray hit: check terrain bounds and ray sensor.')
        self.frames[:, 1:] = self.frames[:, :-1].clone()
        self.frames[:, 0] = frame
        self.frames[self.fresh] = frame[self.fresh, None, :]
        self.fresh[:] = False
        actor = torch.cat((self.frames.flatten(1), heights), -1).clamp(-OBS_CLIP, OBS_CLIP)
        critic = torch.cat((frame, scaled_velocity, force, heights), -1).clamp(-OBS_CLIP, OBS_CLIP)
        return actor, critic
