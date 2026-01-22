# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Image encoding stages for I2V diffusion pipelines.

This module contains implementations of image encoding stages for diffusion pipelines.
"""

import PIL
import torch

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.vaes.common import ParallelTiledVAE
from sglang.multimodal_gen.runtime.models.vision_utils import (
    normalize,
    numpy_to_pt,
    pil_to_numpy,
    resize,
)
from sglang.multimodal_gen.runtime.pipelines.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines.stages.validators import VerificationResult
from sglang.multimodal_gen.runtime.server_args import ExecutionMode, ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)

class FluxPBRVAEEncodingStage(PipelineStage):
    """
    Stage for encoding pixel representations into latent space.

    This stage handles the encoding of pixel representations into the final
    input format (e.g., latents).
    """

    def __init__(self, vae: ParallelTiledVAE, vae_image_processor, vae_scale_factor, **kwargs) -> None:
        super().__init__()
        self.vae = vae
        self.vae_image_processor = vae_image_processor
        self.vae_scale_factor = vae_scale_factor

    @staticmethod
    def _pack_latents(
        latents, batch_size, num_channels_latents, height, width, pixel_shuffle=True
    ):
        if pixel_shuffle:
            latents = latents.view(
                batch_size, num_channels_latents, height // 2, 2, width // 2, 2
            )
            latents = latents.permute(0, 2, 4, 1, 3, 5)
            latents = latents.reshape(
                batch_size, (height // 2) * (width // 2), num_channels_latents * 4
            )
        else:
            latents = latents.permute(0, 2, 3, 1)
            latents = latents.reshape(batch_size, height * width, num_channels_latents)
        return latents
    @staticmethod
    def _prepare_latent_image_ids(
        batch_size, height, width, device, dtype, offset_x=0, offset_y=0, offset_z=0
    ):
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1]
            + torch.arange(offset_y, offset_y + height)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(offset_x, offset_x + width)[None, :]
        )
        if offset_z != 0:
            latent_image_ids[..., 0] = latent_image_ids[..., 0] + offset_z
        latent_image_ids = latent_image_ids.reshape(height * width, 3).to(
            device=device, dtype=dtype
        )
        return latent_image_ids

    @staticmethod
    def retrieve_latents(
        encoder_output: torch.Tensor,
        generator:  None,
        sample_mode: str = "sample",
    ):
        if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
            return encoder_output.latent_dist.sample(generator)
        elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
            return encoder_output.latent_dist.mode()
        elif hasattr(encoder_output, "latents"):
            return encoder_output.latents
        else:
            raise AttributeError("Could not access latents of provided encoder_output")

    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if isinstance(generator, list):
            image_latents = [
                self.retrieve_latents(
                    self.vae.encode(image[i : i + 1]), generator=generator[i]
                )
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = self.retrieve_latents(
                self.vae.encode(image), generator=generator
            )

        image_latents = (
            image_latents - self.vae.config.shift_factor
        ) * self.vae.config.scaling_factor

        return image_latents

    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        control_image = batch.control_image
        add_control_image = batch.add_control_image
        dual_image = batch.ref_image
        HL = 2 * (int(batch.height) // (self.vae_scale_factor * 2))
        WL = 2 * (int(batch.width) // (self.vae_scale_factor * 2))
        if dual_image is not None:
            WD, HD = dual_image.size
            dual_image = self.vae_image_processor.preprocess(
                dual_image, height=HD, width=WD
            )
            dual_image = dual_image.to(dtype=torch.bfloat16, device="cuda")
            dual_latents = self._encode_vae_image(image=dual_image, generator=batch.generator)
            dual_latents = dual_latents.to(dtype=torch.bfloat16)
            BDL, CDL, HDL, WDL = dual_latents.shape
            # VAE applies 8x compression on images but we must also account for packing which requires
            # latent height and width to be divisible by 2.
            assert HDL == 2 * (HD // (self.vae_scale_factor * 2)) and WDL == 2 * (
                WD // (self.vae_scale_factor * 2)
            )
            dual_latents = self._pack_latents(
                dual_latents,
                1,
                dual_latents.shape[1],
                HDL,
                WDL,
                pixel_shuffle=True,
            )
            dual_ids = self._prepare_latent_image_ids(
                batch.batch_size,
                HDL // 2,
                WDL // 2,
                get_local_torch_device(),
                torch.bfloat16,
                offset_x=WL // 2,
                offset_y=HL // 2,
                offset_z=0,
            )
        else:
            dual_latents = None
            dual_ids = None
        if control_image is not None:
            WC, HC = control_image.size
            control_image = self.vae_image_processor.preprocess(
                control_image, height=HC, width=WC
            )
            control_image = control_image.to(dtype=torch.bfloat16, device=get_local_torch_device())
            control_latents = self._encode_vae_image(
                image=control_image, generator=batch.generator
            )
            control_latents = control_latents.to(dtype=torch.bfloat16)
            BCL, CCL, HCL, WCL = control_latents.shape
            # VAE applies 8x compression on images but we must also account for packing which requires
            # latent height and width to be divisible by 2.
            assert HCL == 2 * (HC // (self.vae_scale_factor * 2)) and WCL == 2 * (
                WC // (self.vae_scale_factor * 2)
            )
            control_latents = self._pack_latents(
                control_latents,
                batch.batch_size,
                control_latents.shape[1],
                HCL,
                WCL,
                pixel_shuffle=True,
            )
            control_ids = self._prepare_latent_image_ids(
                batch.batch_size,
                HCL // 2,
                WCL // 2,
                get_local_torch_device(),
                torch.bfloat16,
                offset_x=0,
                offset_y=0,  # HL // 2,
                offset_z=1,
            )
        else:
            control_latents = None
            control_ids = None

        if add_control_image is not None:
            WC, HC = add_control_image.size
            add_control_image = self.vae_image_processor.preprocess(
                add_control_image, height=HC, width=WC
            )
            add_control_image = add_control_image.to(
                dtype=torch.bfloat16, device=get_local_torch_device()
            )
            add_control_latents = self._encode_vae_image(
                image=add_control_image, generator=batch.generator
            )
            add_control_latents = add_control_latents.to(dtype=torch.bfloat16)
            BACL, CACL, HACL, WACL = add_control_latents.shape
            # VAE applies 8x compression on images but we must also account for packing which requires
            # latent height and width to be divisible by 2.
            assert HACL == 2 * (HC // (self.vae_scale_factor * 2)) and WACL == 2 * (
                WC // (self.vae_scale_factor * 2)
            )
            add_control_latents = self._pack_latents(
                add_control_latents,
                batch.batch_size,
                add_control_latents.shape[1],
                HACL,
                WACL,
                pixel_shuffle=True,
            )
            add_control_ids = self._prepare_latent_image_ids(
                batch.batch_size,
                HACL // 2,
                WACL // 2,
                get_local_torch_device(),
                torch.bfloat16,
                offset_x=0,  # 0,
                offset_y=0,  # HACL // 2,
                offset_z=2,  # 0,
            )
        else:
            add_control_latents = None
            add_control_ids = None

        if control_latents is not None and add_control_latents is not None:
            control_latents = torch.cat([control_latents, add_control_latents], dim=1)
            control_ids = torch.cat([control_ids, add_control_ids], dim=0)
        elif control_latents is not None or add_control_latents is not None:
            control_latents = (
                control_latents if control_latents is not None else add_control_latents
            )
            control_ids = control_ids if control_ids is not None else add_control_ids
        else:
            control_latents = None
            control_ids = None
        condition_latents = torch.cat([control_latents, dual_latents], dim=1)
        condition_ids = torch.cat([control_ids, dual_ids], dim=0)
        batch.image_latent = condition_latents
        batch.condition_latents_id = condition_ids
        return batch