# Jobs

This directory contains all jobs used for training or fine-tuning various models.

## StyleGAN3

The [stylegan3](./stylegan3) directory contains:

* `requirements.txt` is used with other module installations to set up the environment for training
[the stylegan3 model](https://github.com/NVlabs/stylegan3) with the Bliss single characters.

* `job_stylegan3.sh` is the job script submitted in [the Cedar platform](https://docs.alliancecan.ca/wiki/Cedar)
to perform the training.

Refer to the [documentation](../docs/TrainStyleGAN3MOdel.md) about how to train this model, training results and
the conclusion about how useful it is.
