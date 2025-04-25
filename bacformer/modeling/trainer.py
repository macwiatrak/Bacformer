from torch.utils.data import Dataset
from transformers import DataCollator, Trainer, TrainingArguments, is_datasets_available

from bacformer.modeling.modeling_base import BacformerModel
from bacformer.modeling.modeling_pretraining import BacformerForCausalProteinFamilyModeling

if is_datasets_available():
    pass


class BacformerTrainer(Trainer):
    """HuggingFace Trainer for Bacformer."""

    def __init__(
        self,
        model: BacformerModel,
        args: TrainingArguments = None,
        data_collator: DataCollator | None = None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            **kwargs,
        )

    def compute_loss(
        self,
        model: BacformerModel,
        inputs: dict,
        num_items_in_batch: int = None,
        return_outputs: bool = False,
    ):
        """Compute loss for Bacformer."""
        # shape [batch_size, seq_len, dim]
        outputs = model(
            protein_embeddings=inputs.pop("protein_embeddings"),
            special_tokens_mask=inputs.pop("special_tokens_mask"),
            token_type_ids=inputs.pop("token_type_ids"),
            attention_mask=inputs.pop("attention_mask"),
            labels=inputs.pop("labels"),
        )

        if return_outputs:
            return outputs[0], outputs[1:]
        return outputs[0]


class BacformerCausalProteinFamilyTrainer(Trainer):
    """HuggingFace Trainer for Bacformer."""

    def __init__(
        self,
        model: BacformerModel,
        args: TrainingArguments = None,
        data_collator: DataCollator | None = None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            **kwargs,
        )

    def compute_loss(
        self,
        model: BacformerForCausalProteinFamilyModeling,
        inputs: dict,
        num_items_in_batch: int = None,
        return_outputs: bool = False,
    ):
        """Compute loss for Bacformer."""
        # shape [batch_size, seq_len, dim]
        outputs = model(
            labels=inputs.pop("labels"),
            special_tokens_mask=inputs.pop("special_tokens_mask"),
            token_type_ids=inputs.pop("token_type_ids"),
            property_ids=inputs.pop("property_ids", None),
        )

        if return_outputs:
            return outputs[0], outputs[1:]
        return outputs[0]
