from typing import Literal

from tap import Tap

from bacformer.modeling.config import SPECIAL_TOKENS_DICT


class BacformerArgumentParser(Tap):
    """Argument parser for training Bacformer."""

    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    # file paths for loading data
    input_dir: str
    output_dir: str

    # model arguments
    batch_size: int = 2
    lr: float = 0.0001
    num_hidden_layers: int = 6
    num_attention_heads: int = 4
    hidden_size: int = 480
    intermediate_size: int = 1280
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 6000
    max_token_type_embeddings: int = 1000
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    protein_clusters_vocab_size: int = 50000  # nr of protein family clusters
    num_labels: int = 1  # nr of labels for classification in downstream tasks
    is_causal_gm: bool = False  # if doing causal language modelling
    return_dict: bool = False  # if returning a dictionary from the model
    return_attn_weights: bool = False  # if returning attention weights from the model
    problem_type: Literal[
        "regression", "single_label_classification", "multi_label_classification", "binary_classification"
    ] = "single_label_classification"  # for downstream tasks
    weight_decay: float = 0.01
    alpha_contrastive_loss: float = 0.5  # for pretraining a model with contrastive loss

    # trainer arguments
    max_epochs: int = 10
    early_stopping_patience: int = 10
    test: bool = False
    random_state: int = 30
    warmup_proportion: float = 0.1  # use warmup for 10% of  all steps
    max_grad_norm: float = 2.0  # for gradient clipping
    gradient_accumulation_steps: int = 8
    logging_steps: int = 100
    monitor_metric: str = "loss"
    dataloader_num_workers: int = 4
    eval_steps: int = 4000
    save_steps: int = 4000

    # data arguments
    mgm_probability: float = 0.15
    max_n_proteins: int = 6000
    max_n_contigs: int = 1000
    special_tokens_dict: dict[str, int] = SPECIAL_TOKENS_DICT
    n_total_samples: int = 1203731  # n total samples in the train set
    n_nodes: int = 1

    pretrained_model_dir: str = None  # "/Users/maciejwiatrak/Downloads/causal-model-ckpt-17798-pretrained/"
