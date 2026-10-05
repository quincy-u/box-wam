# Third-party notices

Box-WAM builds on the following projects. Their licenses apply to the corresponding parts.

## V-JEPA 2 (MIT)

The transformer blocks in `boxwam/model/transformer.py` (rotary attention, its 3-D RoPE segment
layout and the depth-dependent residual rescaling) and the training-loop conventions were
derived from Meta's [V-JEPA 2](https://github.com/facebookresearch/vjepa2) code base, which is
released under the MIT license:

> Copyright (c) Meta Platforms, Inc. and affiliates.
>
> Permission is hereby granted, free of charge, to any person obtaining a copy of this software
> and associated documentation files (the "Software"), to deal in the Software without
> restriction, including without limitation the rights to use, copy, modify, merge, publish,
> distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
> Software is furnished to do so, subject to the following conditions: The above copyright notice
> and this permission notice shall be included in all copies or substantial portions of the
> Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.

## DINOv3 (DINOv3 License)

The frozen image encoder is DINOv3 ViT-L/16 (`vit_large_patch16_dinov3.lvd1689m`), downloaded
through [timm](https://github.com/huggingface/pytorch-image-models) from the Hugging Face Hub.
The weights are distributed by Meta under the DINOv3 license; this repository does not
redistribute them.

## openpi (Apache 2.0)

The multi-task route (`multitask/`) is a fork of Physical Intelligence's
[openpi](https://github.com/Physical-Intelligence/openpi), released under the Apache 2.0 license.
Our fork is `multitask/openpi-boxwam`, which keeps openpi's license files; see `multitask/README.md`.

## Gemma (Gemma Terms of Use)

The pi0.5 base model contains PaliGemma weights, which are derived from Google's Gemma. The released multi-task
checkpoint [`quincyu/pi05_robotwin_obb_eef`](https://huggingface.co/quincyu/pi05_robotwin_obb_eef) is pi0.5
fine-tuned on RoboTwin 2.0 with the Box-WAM head, so it is a Model Derivative of Gemma and its use and
redistribution are subject to the Gemma Terms of Use, including the use restrictions of its Section 3.2 and the
[Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy). A copy of the terms is in
`multitask/openpi-boxwam/LICENSE_GEMMA.txt` and is distributed with the checkpoint.

> Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms

The standalone Box-WAM policies (`quincyu/box-wam-standalone`) do not contain Gemma weights.

## RoboTwin 2.0 (MIT)

Evaluation uses a fork of [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin):
`eval/RoboTwin-boxwam`, which keeps RoboTwin's license file.
