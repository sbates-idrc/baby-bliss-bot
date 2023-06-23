# Jobs
This contains all jobs used for training or fine-tuning various models.

## StyleGAN2-ADA
The [stylegan2-ada](./stylegan2-ada) folder contains:

   - `def-styleGan2AdaPytorchDataSetupBatch.sh` is the SBatch script for preparing the training dataset for StyleGAN2-ADA.  The script uses the `def-whkchun` cluster.
   - `def-styleGAN2AdaPytorchTrainBatch.sh` is the SBatch script for training. The script uses the `def-whkchun` cluster.
   - `ctb-styleGAN2AdaPytorchGenerateBatch.sh` is the SBatch script for generating an image from the StyleGAN2-ADA model.  The script uses the `ctb-whkchun` cluster.
   - `def-styleGAN2AdaPytorchGenerateBatch.sh` is the SBatch script that also can be used to generate images from the StyleGAN2-ADA model.  This version uses the `def-whkchun` cluster.
   - `requirements.txt` shows the packages used by the PyTorch implementation of StyleGAN2-ADA.  Note that this is not used to create the environment, but to document the environment after it was created.

See the [StyleGAN2-ADATraining.md](../docs/StyleGAN2-ADATraining.md) in the [documentation](../docs) folder for details on how to set up the environment.
