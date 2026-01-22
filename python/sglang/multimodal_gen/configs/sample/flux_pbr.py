# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
import torch

from sglang.multimodal_gen.configs.sample.base import SamplingParams


@dataclass
class FluxPBRSamplingParams(SamplingParams):
    # Video parameters
    # height: int = 1024
    # width: int = 1024
    num_frames: int = 1
    # Denoising stage
    guidance_scale: float = 3.5
    negative_prompt: str = None
    # latents: list[torch.Tensor] = None
    # num_inference_steps: int = 50
