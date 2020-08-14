import torch
import runway
import numpy as np


@runway.setup(options={"checkpoint": runway.category(description="Pretrained checkpoints to use.",
                                       choices=['celebAHQ-512', 'celebAHQ-256', 'celeba'],
                                       default='celebAHQ-512')})
def setup(opts):
    checkpoint = opts['checkpoint']
    use_gpu = True if torch.cuda.is_available() else False
    # Load the model from the Pytorch Hub
    model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
                          'PGAN', model_name=checkpoint,
                           pretrained=True, useGPU=use_gpu)
   return model

+ @runway.command('generate',
+               inputs={ 'z': runway.vector(length=512, sampling_std=0.5)},
+               outputs={ 'image': runway.image })
+  def generate(model, inputs):
+  # Generate ♾ infinite ♾ images
+   z = inputs['z']
+   latents = z.reshape((1, 559))
+   latents = torch.from_numpy(latents)
-   noise, _ = model.buildNoiseData(1)
    with torch.no_grad():
+       generated_image = model.test(latents.float())
+   generated_image = generated_image.clamp(min=-1, max=1)
+   generated_image = ((generated_image + 1.0) * 255 / 2.0)
    # Now generated_image contains our generated image! 🌞
+   return generated_image[0].permute(1, 2, 0).numpy().astype(np.uint8)

+ if __name__ == '__main__':
+    runway.run(port=5232)
