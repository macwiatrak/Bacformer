from collections import OrderedDict

import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy, mse_loss

from bacformer.modeling.config import BacformerConfig
from bacformer.modeling.modeling_base import BacformerModel, BacformerModelOutput, BacformerPreTrainedModel


class BacformerForProteinClassification(BacformerPreTrainedModel):
    """Bacformer model with a classification head on top for protein classification tasks."""

    def __init__(self, config: BacformerConfig, benchmark_esm: bool = False):
        super().__init__(config)
        self.config = config
        self.benchmark_esm = benchmark_esm

        self.bacformer = BacformerModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        protein_embeddings: torch.Tensor,
        special_tokens_mask: torch.Tensor,
        labels: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        return_attn_weights: bool = None,
        return_dict: bool | None = None,
    ) -> BacformerModelOutput | None:
        """Forward method for the model."""
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        return_attn_weights = (
            return_attn_weights if return_attn_weights is not None else self.config.return_attn_weights
        )

        if self.benchmark_esm:
            outputs = [protein_embeddings]
        else:
            outputs = self.bacformer(
                protein_embeddings=protein_embeddings,
                special_tokens_mask=special_tokens_mask,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                return_attn_weights=return_attn_weights,
                return_dict=return_dict,
            )

        last_hidden_state = outputs[0]

        last_hidden_state = self.dropout(last_hidden_state)
        logits = self.classifier(last_hidden_state)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)

            if self.config.problem_type == "regression":
                loss = mse_loss(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss = cross_entropy(logits.view(-1, self.config.num_labels), labels.view(-1))
            elif (
                self.config.problem_type == "multi_label_classification"
                or self.config.problem_type == "binary_classification"
            ):
                # remove the -100 labels from loss computation
                mask = torch.ones_like(labels.view(-1)) - (labels.view(-1) == -100.0).float()
                loss = binary_cross_entropy_with_logits(
                    logits.view(-1), labels.view(-1).type_as(logits), reduction="none"
                )
                loss = (loss * mask).sum() / mask.sum()

        if not return_dict:
            return (
                loss,
                None,
                logits,
            )

        return BacformerModelOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=last_hidden_state,
            attentions=outputs.attentions,
        )


class BacformerForGenomeClassification(BacformerPreTrainedModel):
    """Bacformer model with a classification head on top for genome classification tasks."""

    def __init__(self, config: BacformerConfig):
        super().__init__(config)
        self.config = config

        self.bacformer = BacformerModel(config, add_pooling_layer=False)
        self.classifier = BacformerGenomeClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        protein_embeddings: torch.Tensor,
        special_tokens_mask: torch.Tensor,
        labels: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        return_attn_weights: bool = None,
        return_dict: bool | None = None,
    ) -> BacformerModelOutput | None:
        """Forward method for the model."""
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        return_attn_weights = (
            return_attn_weights if return_attn_weights is not None else self.config.return_attn_weights
        )

        outputs = self.bacformer(
            protein_embeddings=protein_embeddings,
            special_tokens_mask=special_tokens_mask,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            return_attn_weights=return_attn_weights,
            return_dict=return_dict,
        )
        last_hidden_state = outputs[0]
        logits = self.classifier(last_hidden_state, attention_mask)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)

            if self.config.problem_type == "regression":
                loss = mse_loss(logits.view(-1), labels.view(-1))
            elif self.config.problem_type == "binary_classification":
                loss = binary_cross_entropy_with_logits(logits.view(-1), labels.view(-1).type_as(logits))
            elif self.config.problem_type == "single_label_classification":
                loss = cross_entropy(logits.view(-1, self.config.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss = binary_cross_entropy_with_logits(logits, labels)

        if not return_dict:
            return (
                loss,
                None,
                logits,
            )

        return BacformerModelOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=outputs.last_hidden_state,
            attentions=outputs.attentions,
        )


class BacformerForProteinProteinInteraction(BacformerPreTrainedModel):
    """Bacformer model with a protein-protein interaction head on top."""

    def __init__(self, config: BacformerConfig):
        super().__init__(config)
        self.config = config
        self.return_attn_weights = config.return_attn_weights

        self.bacformer = BacformerModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
            nn.Dropout(0.2),
        )
        self.ppi_head = BacformerProteinProteinInteractionHead(
            in_features=config.hidden_size, prot_emb_idx=config.prot_emb_token_id
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        protein_embeddings: torch.Tensor,
        special_tokens_mask: torch.Tensor,
        labels: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        return_attn_weights: bool = None,
        return_dict: bool | None = None,
    ) -> OrderedDict | None:
        """Forward method for the model."""
        assert protein_embeddings.shape[0] == 1, "Batch size should be 1 for protein-protein interaction task"
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        outputs = self.bacformer(
            protein_embeddings=protein_embeddings,
            special_tokens_mask=special_tokens_mask,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            return_attn_weights=False,
            return_dict=True,
        )

        outputs.last_hidden_state = outputs.last_hidden_state.squeeze(0)[1:-2, :]
        outputs.last_hidden_state = self.dense(self.dropout(outputs.last_hidden_state))
        outputs.last_hidden_state = torch.stack(
            [outputs.last_hidden_state[labels[:, 0]], outputs.last_hidden_state[labels[:, 1]]]
        ).mean(dim=0)
        logits = self.ppi_head(outputs.last_hidden_state)

        loss = binary_cross_entropy_with_logits(logits, labels[:, 2].type_as(logits).squeeze(0))

        if not return_dict:
            return (
                loss,
                logits,
            )

        return BacformerModelOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=outputs.last_hidden_state,
            attentions=outputs.attentions,
        )


class BacformerGenomeClassificationHead(nn.Module):
    """Head for genome-level classification tasks."""

    def __init__(self, config: BacformerConfig):
        super().__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features: torch.Tensor, padding_mask: torch.Tensor, **kwargs):
        """Forward method for the head."""
        if padding_mask is not None:
            x = torch.einsum("ijk,ij->ik", features, padding_mask) / padding_mask.sum(1).unsqueeze(1)
        else:
            x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class BacformerProteinProteinInteractionHead(nn.Module):
    """Head for protein-protein interaction task at a genome level."""

    def __init__(self, in_features: int, prot_emb_idx: int = 4, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.prot_emb_idx = prot_emb_idx
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(in_features, 1, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward method for the head."""
        return self.linear(self.dropout(hidden_states)).squeeze(-1)
