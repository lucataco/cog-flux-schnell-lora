# FLUX.1-Schnell LoRA Explorer Cog Model

This is an implementation of [black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) as a [Cog](https://github.com/replicate/cog) model.

Named LoRA Explorer, to explore the model with different LoRA weights.

## Development

Follow the [model pushing guide](https://replicate.com/docs/guides/push-a-model) to push your own model to [Replicate](https://replicate.com).


## How to use

Make sure you have [cog](https://github.com/replicate/cog) installed.

To run a prediction:

    cog predict -i prompt="a beautiful castle frstingln illustration" -hf_lora="alvdansen/frosting_lane_flux"

![Output](output.0.png)
