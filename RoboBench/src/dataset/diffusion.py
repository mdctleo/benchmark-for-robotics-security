from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
from diffusers.utils import numpy_to_pil
from tqdm import tqdm
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


class Diffusion:
    """
    Text-to-image and image-edit wrapper around the vLLM-Omni ``Omni`` engine.

    Public API:
    - `create_images(prompts, save_paths, ...)` writes images to disk
    - `edit_images(prompts, image_paths, save_paths)` writes images to disk
    """

    def __init__(
        self,
        model_name,
        num_inference_steps=4,
        guidance_scale=1.0,
        num_images_per_prompt=1,
    ):
        self.model_name = model_name
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.num_images_per_prompt = num_images_per_prompt

        self.backend_name = "vllm-omni"
        self._closed = False

        self.engine = self._load_engine()

    def _load_engine(self):
        model_id = self._resolve_model_id()
        tensor_parallel_size = max(torch.cuda.device_count(), 1)

        engine = Omni(
            model=model_id,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            cache_backend=os.getenv("DIFFUSION_OMNI_CACHE_BACKEND") or None,
            vae_use_slicing=self._env_flag("DIFFUSION_OMNI_VAE_SLICING", default=False),
            vae_use_tiling=self._env_flag("DIFFUSION_OMNI_VAE_TILING", default=False),
            enforce_eager=self._env_flag("DIFFUSION_OMNI_ENFORCE_EAGER", default=False),
        )

        print(
            f"Initialized {self.model_name} with {self.backend_name}"
            f" using model={model_id}, tp={tensor_parallel_size}",
            flush=True,
        )
        return engine

    def _resolve_model_id(self):
        if self.model_name == "flux2-dev":
            return "black-forest-labs/FLUX.2-dev"
        if self.model_name == "flux2-klein-fast":
            return "black-forest-labs/FLUX.2-klein-4B"
        if self.model_name == "flux2-klein":
            return "black-forest-labs/FLUX.2-klein-9B"

        raise ValueError(f"Unknown model name: {self.model_name}")

    def _env_flag(self, name, default=False):
        value = os.getenv(name)
        if value is None:
            return default
        return value.strip().lower() not in {"0", "false", "no", "off"}

    def _generate_numbered_paths(self, save_path, count):
        """Generate numbered file paths when multiple images are produced."""
        if count == 1:
            return [save_path]

        path = Path(save_path)
        stem = path.stem
        suffix = path.suffix
        parent = path.parent

        return [str(parent / f"{stem}_{i}{suffix}") for i in range(count)]

    def _save_images(self, images, save_paths):
        for image, save_path in zip(images, save_paths):
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            image.save(save_path)

    def _build_sampling_params(self, *, height=None, width=None, max_sequence_length=None, seed=None):
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        return OmniDiffusionSamplingParams(
            seed=seed,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            num_outputs_per_prompt=self.num_images_per_prompt,
            height=height,
            width=width,
            max_sequence_length=max_sequence_length,
        )

    def _build_text_prompt_payload(self, prompt):
        return {"prompt": prompt, "modalities": ["image"]}

    def _build_edit_prompt_payload(self, prompt, image_path):
        return {
            "prompt": prompt,
            "modalities": ["image"],
            "multi_modal_data": {"image": str(Path(image_path))},
        }

    def _generate(self, payload, sampling_params):
        outputs = self.engine.generate(payload, sampling_params_list=[sampling_params])
        if not outputs:
            raise RuntimeError("No output returned from the diffusion engine.")
        images = getattr(outputs[0], "images", None)
        if not images:
            raise RuntimeError("No images returned from the diffusion engine.")
        normalized = []
        for image in images:
            if isinstance(image, np.ndarray):
                normalized.extend(numpy_to_pil(image))
            else:
                normalized.append(image)
        return normalized

    def create_images(
        self,
        prompts,
        save_paths,
        height=1024,
        width=1024,
        max_sequence_length=512,
        seeds=None,
    ):
        """
        Args:
            prompts (list of str): A list of text prompts.
            save_paths (list of str): A list of file paths to save the images.
            height, width (int): Output size; FLUX.2 requires multiples of 16.
            max_sequence_length (int): Text encoder sequence length.
            seeds (list of int or None): One seed per prompt; random when None.

        Returns:
            None. Images are written to `save_paths`.
        """
        if len(prompts) != len(save_paths):
            raise ValueError("The number of prompts must match the number of save paths.")
        if seeds is None:
            seeds = [None] * len(prompts)
        if len(seeds) != len(prompts):
            raise ValueError("The number of seeds must match the number of prompts.")

        # vLLM-Omni sampling params are per stage, not per prompt, so each image is
        # its own request to keep per-image seeds.
        for prompt, save_path, seed in tqdm(
            zip(prompts, save_paths, seeds),
            desc="Generating images...",
            total=len(prompts),
        ):
            sampling_params = self._build_sampling_params(
                height=height,
                width=width,
                max_sequence_length=max_sequence_length,
                seed=seed,
            )
            images = self._generate(self._build_text_prompt_payload(prompt), sampling_params)
            save_targets = self._generate_numbered_paths(save_path, len(images))
            self._save_images(images, save_targets)

    def edit_images(self, prompts, image_paths, save_paths):
        """
        Args:
            prompts (list of str): A list of text prompts describing the desired edits.
            image_paths (list of str): A list of image paths.
            save_paths (list of str): A list of file paths to save the edited images.

        Returns:
            None. Images are written to `save_paths`.

        Note: ported to the sync engine alongside `create_images` but not exercised by
        the dataset driver; the `multi_modal_data` payload is unchanged from before.
        """
        if len(prompts) != len(image_paths):
            raise ValueError("The number of prompts must match the number of images.")

        if len(image_paths) != len(save_paths):
            raise ValueError("The number of image paths must match the number of save paths.")

        for prompt, image_path, save_path in tqdm(
            zip(prompts, image_paths, save_paths),
            desc="Editing images...",
            total=len(prompts),
        ):
            sampling_params = self._build_sampling_params()
            images = self._generate(self._build_edit_prompt_payload(prompt, image_path), sampling_params)
            save_targets = self._generate_numbered_paths(save_path, len(images))
            self._save_images(images, save_targets)

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self.engine is not None:
            self.engine.close()

    def __del__(self):  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except Exception:
            pass
