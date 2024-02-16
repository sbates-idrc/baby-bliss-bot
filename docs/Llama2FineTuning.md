# Experiment with Llama2 model

The experiment leveraged the [7B parameter Llama2 model pretrained by Meta](https://huggingface.co/meta-llama/Llama-2-7b-hf),
converted for the seamless use of the Hugging Face Transformers format. This model is choosen as a starting
point because it requires less training time and GPU resources compared to its larger counterparts, while it
potentially sacrifies some capability. Additionally, the Hugging Face Transformers format is selected because
of its extensive community support and standardized APIs.

## Download Llama-2-7b-hf to Cedar

1. Request access to Llama2 models on [the Meta website](https://llama.meta.com/llama-downloads/);
2. Followed the instructions on [the Hugging Face website](https://huggingface.co/meta-llama/Llama-2-7b-hf)
to request the access to its Llama2 model;
3. Request a hugging face access token on [this page](https://huggingface.co/settings/tokens);
4. Login to the Cedar cluster;
5. Create a "llama" directory and run these commands to download the model:

```
mkdir llama2
cd llama2

// Load git-lfs first for downloading via Git large file storage
module load git-lfs/3.3.0
git lfs install

git clone https://{hugging_face_id}:{hugging_face_access_token}@huggingface.co/meta-llama/Llama-2-7b-hf

// Fetch git large files in the repo directory
cd Llama-2-7b-hf
git lfs fetch
```

6. Copy the content of [`requirements.txt`](https://github.com/facebookresearch/llama/blob/main/requirements.txt)
for setting up the Llama2 models into a new file named `requirements-llama2.txt` in the "llama" directory.

## Use the Llama2 model

In the [`jobs/original_use`](../jobs/Llama2/original_use) directory, there are two scripts:

* original_use_7b_hf.py: The script that loads the downloaded model and tokenizer to perform text generation,
word predictions and making inferences
* job_original_use_7b_hf.sh: The job script submitted to Cedar to run `original_use_7b_hf.py`

Note that the job script must be copied to the user's `scratch` directory and is submitted from there using
the `sbatch` command.

FTP scripts above to the cedar cluster in the users `llama2/original_use` directory. Run the following command to
submit the job.

```
cp llama2/original_use/job_original_use_7b_hf.sh scratch/.
cd scratch
sbatch job_original_use_7b_hf.sh
```

The result is written to the `llama2/original_use/result.txt`.

## Fine-tune the Llama2 model

In the [`jobs/finetune`](../jobs/Llama2/finetune) directory, there are these scripts:

* bliss.json: The dataset that converts English text to the structure in the Bliss language
* finetune_7b_hf.py: The script that fine-tunes the downloaded model
* job_finetune_7b_hf.sh: The job script submitted to Cedar to run `finetune_7b_hf.py`

FTP scripts above to the cedar cluster in the users `llama2/finetune` directory. Run the following command to
submit the job.

```
cp llama2/finetune/job_finetune_7b_hf.sh scratch/.
cd scratch
sbatch job_finetune_7b_hf.sh
```

## References

[Llama2 in the Facebook Research Github repository](https://github.com/facebookresearch/llama)
[Llama2 fine-tune, inference examples](https://github.com/facebookresearch/llama-recipes)
[Llama2 on Hugging Face](https://huggingface.co/docs/transformers/model_doc/llama2)
[Use Hugging Face Models on Cedar Clusters](https://docs.alliancecan.ca/wiki/Huggingface)
