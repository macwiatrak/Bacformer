# Bacformer

[//]: # ([![Tests][badge-tests]][tests])

[//]: # ([![Documentation][badge-docs]][documentation])

[//]: # ()
[//]: # ([badge-tests]: https://img.shields.io/github/actions/workflow/status/macwiatrak/Bacformer/test.yaml?branch=main)

[//]: # ([badge-docs]: https://img.shields.io/readthedocs/Bacformer)

Bacformer is a prokaryotic foundational model which models
whole-bacterial genomes as a sequence of proteins ordered by their genomic coordinates on the chromosome and plasmid(s).
It takes as input average protein embeddings from protein language models and computes contextualised protein
embeddings conditional on other proteis present in the genome. Bacformer is trained on a diverse dataset of ~1.3M bacterial genomes and ~3B proteins.

![Bacformer](files/Bacformer.png)

Bacformer can be applied to a wide range of tasks, including: strain clustering, essential genes prediction, operon identification,
ppi prediction, protein function prediction and more. We provide [model checkpoints]() for pretrained models as well as Bacformer
finetuned for various tasks. We also provide tutorials and make Bacformer available via [HuggingFace](https://huggingface.co/macwiatrak).

## News

- **2025-05-15**: Bacformer is now available on [HuggingFace](https://huggingface.co/macwiatrak).

## Contents

- [Setup](#setup)
  - [Requirements](#requirements)
  - [installation](#installation)
- [Usage](#usage)
  - [Tutorials](#tutorials)
- [HuggingFace](#huggingface)
- [Pretrained model checkpoints](#pretrained-model-checkpoints)
- [Contributing](#contributing)
- [Citation](#citation)
- [Installation](#installation)
- [Release notes](#release-notes)
- [Contact](#contact)

## Setup

### Requirements

Bacformer is based on [PyTorch](https://pytorch.org/) and [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)
and was developed in `python=3.10`.

Bacformer uses [ESM-2](https://github.com/facebookresearch/esm) protein embedding as input (`esm2_t12_35M_UR50D`). We
recommend using the [faplm](https://github.com/pengzhangzhi/faplm) package to compute protein embeddings in a fast and efficient way.

### Installation

[//]: # (You can install Bacformer using `pip`)

[//]: # (```bash)

[//]: # (pip install bacformer)

[//]: # (```)

[//]: # ()
[//]: # (by cloning the repository and installing the dependencies:)

You can install Bacformer by cloning the repository and installing the dependencies.
```bash
git clone https://github.com/macwiatrak/Bacformer.git
cd Bacformer
# 1) install Bacformer **with its core dependencies**
pip install .
# 2) (optional but recommended) add the fast‐attention extra (“faesm”)
pip install ".[faesm]"
```

## Usage

Below is an example of how to use Bacformer to compute contextual protein embeddings.

```python
import torch
from bacformer.modeling import BacformerModel
from bacformer.pp import protein_seqs_to_bacformer_inputs

device = "cuda:0"
model = BacformerModel.from_pretrained("macwiatrak/bacformer-causal-MAG").to(device).eval().to(torch.bfloat16)

# Example input: a sequence of protein sequences
# in this case: 4 toy protein sequences
# Bacformer was trained with a maximum nr of proteins of 6000.
protein_sequences = [
    "MGYDLVAGFQKNVRTI",
    "MKAILVVLLG",
    "MQLIESRFYKDPWGNVHATC",
    "MSTNPKPQRFAWL",
]
# embed the proteins with ESM-2 to get average protein embeddings
inputs = protein_seqs_to_bacformer_inputs(
    protein_sequences,
    device=device,
    batch_size=128,  # the batch size for computing the protein embeddings
    max_n_proteins=6000,  # the maximum number of proteins Bacformer was trained with
)

# move the inputs to the device
inputs = {k: v.to(device) for k, v in inputs.items()}
# compute contextualized protein embeddings with Bacformer
with torch.no_grad():
    outputs = model(**inputs, return_dict=True)

print('last hidden state shape:', outputs["last_hidden_state"].shape)  # (batch_size, max_length, hidden_size)
print('genome embedding:', outputs.last_hidden_state.mean(dim=1).shape)  # (batch_size, hidden_size)
```

### Tutorials

We provide a set of tutorials to help you get started with Bacformer. The tutorials cover the following topics:
- [Bacformer for strain clustering](tutorials/strain_clustering_tutorial.ipynb)
- [Finetuning Bacformer for essential genes prediction](tutorials/gene_essentiality_prediction_tutorial.ipynb)
- [Bacformer for phenotypic traits prediction](tutorials/phenotypic_traits_prediction_tutorial.ipynb)
- [Finetuning Bacformer for phenotypic traits prediction](tutorials/finetune_phenotypic_traits_prediction_tutorial.ipynb)

We are actively working on more tutorials, so stay tuned! If you have any suggestions for tutorials, please let us know by raising an issue in the [issue tracker][issue tracker].

## HuggingFace

Bacformer is integrated with [HuggingFace](https://huggingface.co/macwiatrak).

```python
import torch
from transformers import AutoModel, AutoModelForMaskedLM, AutoModelForCausalLM

# load the Bacformer model trained with an autoregressive objective
causal_model = AutoModelForCausalLM.from_pretrained("macwiatrak/bacformer-causal-MAG", trust_remote_code=True).to(torch.bfloat16).eval()

# load the Bacformer model trained with a masked objective
masked_model = AutoModelForMaskedLM.from_pretrained("macwiatrak/bacformer-masked-MAG", trust_remote_code=True).to(torch.bfloat16).eval()

# load the Bacformer encoder model finetuned on complete genomes (i.e. without the protein family classification head)
# we recommend using this model for complete genomes as a start for finetuning on your own dataset for all tasks except generation
encoder_model = AutoModel.from_pretrained("macwiatrak/bacformer-masked-complete-genomes", trust_remote_code=True).to(torch.bfloat16).eval()
```

## Pretrained model checkpoints

We provide a range of pretrained model checkpoints for Bacformer which are available via [HuggingFace](https://huggingface.co/macwiatrak).

| Checkpoint name                                                         | Genome type                         | Description                                                                                                                                                                                                                                                         |
|-------------------------------------------------------------------------|-------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `bacformer-causal-MAG`                                                  | MAG                                 | A model pretrained on the ~1.3 M metagenome-assembled genomes (MAG) with an autoregressive objective.                                                                                                                                                               |
| `bacformer-masked-MAG`                                                  | MAG                                 | A model pretrained on the ~1.3 M metagenome-assembled genomes (MAG) with a masked objective, randomly masking 15 % of proteins.                                                                                                                                     |
| `bacformer-causal-complete-genomes`                                     | Complete (i.e. uninterrupted)       | A `bacformer-causal-MAG` finetuned on a set of ~40 k complete genomes with an autoregressive objective.                                                                                                                                                             |
| `bacformer-masked-complete-genomes`                                     | Complete (i.e. uninterrupted)       | A `bacformer-masked-MAG` finetuned on a set of ~40 k complete genomes with a masked objective, randomly masking 15 % of the proteins.                                                                                                                               |
| `bacformer-causal-protein-family-modeling-complete-genomes`             | Complete (i.e. uninterrupted)       | A `bacformer-causal-MAG` finetuned on a set of ~40 k complete genomes with an autoregressive objective. In contrast to other models, this model takes as input a protein-family token rather than the protein sequence, allowing generation of sequences of protein families. |
| `bacformer-for-essential-genes-prediction`                              | Complete (i.e. uninterrupted)       | A `bacformer-masked-complete-genomes` finetuned on the essential-genes prediction task.                                                                                                                                                                             |
| `bacformer-for-ppi-prediction`                                          | Complete (i.e. uninterrupted)       | A `bacformer-masked-complete-genomes` finetuned on the protein–protein interaction prediction task.                                                                                                                                                                 |



## Contributing

We welcome contributions to Bacformer! If you would like to contribute, please follow these steps:
1. Fork the repository.
2. Install `pre-commit` and set up the pre-commit hooks (make sure to do it at the root of the repository).
```bash
pip install pre-commit
pre-commit install
```
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them.
4. Push your changes to your forked repository.
5. Create a pull request to the main repository.
6. Make sure to add tests for your changes and run the tests to ensure everything is working correctly.

## Contact

For questions, bugs, and feature requests, please raise an issue in the repository.

## Citation

> t.b.a

[uv]: https://github.com/astral-sh/uv
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/macwiatrak/Bacformer/issues
[tests]: https://github.com/macwiatrak/Bacformer/actions/workflows/test.yaml
[documentation]: https://bacformer.readthedocs.io
[changelog]: https://bacformer.readthedocs.io/en/latest/changelog.html
[api documentation]: https://bacformer.readthedocs.io/en/latest/api.html
[pypi]: https://pypi.org/project/bacformer
