from torch.utils.data import Dataset
from transformers import DataCollator, Trainer, TrainingArguments, is_datasets_available

from bacformer.modeling.modeling_base import BacformerModel
from bacformer.modeling.modeling_pretraining import BacformerForCausalProteinFamilyModeling
# For Bacformer Large For Genome Classification
from bacformer.modeling.modeling_large import BacformerLargeForGenomeClassification

if is_datasets_available():
    pass


def _extract_loss_and_logits(outputs):
    """Normalize Bacformer outputs to (loss, logits) across tuple/dict/ModelOutput. Currently used by BacformerLargeForGenomeClassification, but can be used by BacformerForGenomeClassification."""
    if isinstance(outputs, tuple):
        # Typical tuple form: (loss, None, logits)
        return outputs[0], outputs[-1]
    if isinstance(outputs, dict):
        return outputs["loss"], outputs["logits"]
    return outputs.loss, outputs.logits


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

class BacformerLargeTrainer(Trainer):
    """Trainer for BacformerLargeForGenomeClassification.

    Bacformer Large uses contig_ids and attention_mask (no special_tokens_mask/token_type_ids).
    The default BacformerTrainer is for the base 26M model which expects different inputs.
    """

    def __init__(
        self,
        model: BacformerLargeForGenomeClassification,
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
        model: BacformerLargeForGenomeClassification,
        inputs: dict,
        num_items_in_batch: int = None,
        return_outputs: bool = False,
    ):
        """Compute loss for Bacformer Large genome classification."""
        protein_embeddings = inputs.pop("protein_embeddings")
        labels = inputs.pop("labels")
        attention_mask = inputs.pop("attention_mask", None)
        contig_ids = inputs.pop("contig_ids", inputs.pop("token_type_ids", None))
        outputs = model(
            protein_embeddings=protein_embeddings,
            labels=labels,
            attention_mask=attention_mask,
            contig_ids=contig_ids,
        )
        loss, logits = _extract_loss_and_logits(outputs)
        if return_outputs:
            return loss, {"logits": logits}
        return loss

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
