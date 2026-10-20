# Third-party components

This repository's project code is provided under [LICENSE](LICENSE).
Dependencies and separately downloaded pretrained weights keep their own licenses.

- [PyTorch](https://github.com/pytorch/pytorch/blob/main/LICENSE): BSD-style.
- [Diffusers](https://github.com/huggingface/diffusers/blob/main/LICENSE) and
  [Accelerate](https://github.com/huggingface/accelerate/blob/main/LICENSE): Apache-2.0.
- The separately downloaded
  [stabilityai/sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)
  model card declares MIT. The VAE is not included in source/wheel archives;
  preserve its upstream model card and license when redistributing its weights.

The colon and liver image datasets are not distributed by this project.
Obtain them from their original provider and follow that provider's terms.
The split manifest contains ROI/tile identifiers and hashes only.
