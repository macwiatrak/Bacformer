import torch
from torch import nn
from torch.nn.functional import cross_entropy, gelu, softmax

from bacformer.modeling.config import SPECIAL_TOKENS_DICT, BacformerConfig
from bacformer.modeling.modeling_base import (
    BacformerModel,
    BacformerModelOutput,
    BacformerPreTrainedModel,
    BacformerProteinFamilyEmbeddings,
)
from bacformer.modeling.utils import compute_contrastive_loss, top_k_filtering, top_p_filtering


class BacformerGMHead(nn.Module):
    """Bacformer Head for genomic modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # add 1 to the condfig.protein_clusters_vocab_size to account for the end token
        self.decoder = nn.Linear(config.hidden_size, config.protein_clusters_vocab_size + 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.protein_clusters_vocab_size + 1))

    def forward(self, features, **kwargs):
        """Forward method for the head."""
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to nr of labels with bias
        x = self.decoder(x) + self.bias
        return x


class BacformerForCausalGM(BacformerPreTrainedModel):
    """Bacformer model for causal genomic modeling.

    This model is used for pretraining a model to predict the next protein family in a sequence of proteins.
    The model takes as input a sequence of protein embeddings and predicts the next protein family in the sequence.
    """

    _tied_weights_keys = ["gm_head.decoder.weight"]

    def __init__(self, config: BacformerConfig):
        super().__init__(config)
        self.config = config

        self.bacformer = BacformerModel(config, add_pooling_layer=False)
        self.gm_head = BacformerGMHead(config)

        # Initialize weights
        self.init_weights()

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
            attention_mask=None,  # attention mechanism handles the causal mask
            return_attn_weights=return_attn_weights,
            return_dict=return_dict,
            is_causal=True,
        )
        last_hidden_state = outputs[0]
        prediction_scores = self.gm_head(last_hidden_state)

        loss = None
        if labels is not None:
            labels = labels.to(prediction_scores.device)

            shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous().view(-1, prediction_scores.shape[-1])
            labels = labels[:, 1:].contiguous().view(-1)
            loss = cross_entropy(shifted_prediction_scores, labels)

        if not return_dict:
            return (
                loss,
                prediction_scores,
            ) + outputs

        return BacformerModelOutput(
            loss=loss,
            logits=prediction_scores,
            last_hidden_state=outputs.last_hidden_state,
            attentions=outputs.attentions,
        )


class BacformerForMaskedGM(BacformerPreTrainedModel):
    """Bacformer model for masked genomic modeling.

    This model is used for pretraining a model to predict the masked protein families in a sequence of proteins.
    The model takes as input a sequence of protein embeddings, the proteins
    are masked at random (15% of the time) and the model is trained to predict the masked protein families.
    """

    _tied_weights_keys = ["gm_head.decoder.weight"]

    def __init__(self, config: BacformerConfig):
        super().__init__(config)
        self.config = config

        self.bacformer = BacformerModel(config, add_pooling_layer=False)
        self.gm_head = BacformerGMHead(config)

        # Initialize weights
        self.init_weights()

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

        # to speed up the forward pass, let's only consider the masked tokens

        loss = None
        if labels is not None:
            # to speed up the forward pass, let's only consider the masked tokens
            last_hidden_state = last_hidden_state[labels != -100]
            prediction_scores = self.gm_head(last_hidden_state)
            labels = labels.to(prediction_scores.device)

            ### notes
            # use the labels to get -100 for non-masked tokens
            # do not use special_tokens_mask
            # check how the labels are constructed

            # only considering the masked tokens
            labels = labels[labels != -100]
            loss = cross_entropy(prediction_scores, labels)
        else:
            prediction_scores = self.gm_head(last_hidden_state)

        if not return_dict:
            return (
                loss,
                prediction_scores,
            ) + outputs

        return BacformerModelOutput(
            loss=loss,
            logits=prediction_scores,
            last_hidden_state=outputs.last_hidden_state,
            attentions=outputs.attentions,
        )


class BacformerForCausalProteinFamilyModeling(BacformerPreTrainedModel):
    """Bacformer model for causal modeling of protein families.

    Using protein family as tokens rather than protein embeddings. The model takes as input a sequence of protein families,
    where each protein family is represented by an integer transformed into a multidimensional embedding.
    This model loses amino-acid resolution by using protein families as tokens, however, it allows for genome generation.
    """

    _tied_weights_keys = ["gm_head.decoder.weight"]

    def __init__(
        self,
        config: BacformerConfig,
        n_conditional_properties: int = None,
        initialise_from_non_pfm_model: bool = False,
    ):
        super().__init__(config)
        self.config = config
        self.cls_token_id = SPECIAL_TOKENS_DICT["CLS"]

        self.bacformer = BacformerModel(config, add_pooling_layer=False)
        self.gm_head = BacformerGMHead(config)

        if initialise_from_non_pfm_model:
            # Initialize weights
            self.init_weights()
            # overwrite the embeddings with the pretrained
            # protein family embeddings from the decoder of the GM Head
            self.bacformer.embeddings = BacformerProteinFamilyEmbeddings(
                config,
                protein_family_embeddings=self.gm_head.decoder.weight,
                token_type_embeddings=self.bacformer.embeddings.token_type_embeddings.weight,
                special_tokens_embeddings=self.bacformer.embeddings.special_tokens_embeddings.weight,
                n_conditional_properties=n_conditional_properties,
            )
        else:
            self.bacformer.embeddings = BacformerProteinFamilyEmbeddings(
                config,
                n_conditional_properties=n_conditional_properties,
            )
            self.init_weights()

    def forward(
        self,
        labels: torch.Tensor = None,
        special_tokens_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        property_ids: torch.Tensor = None,
        return_attn_weights: bool = None,
        return_dict: bool | None = None,
    ) -> BacformerModelOutput | None:
        """Forward method for the model."""
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        return_attn_weights = (
            return_attn_weights if return_attn_weights is not None else self.config.return_attn_weights
        )

        outputs = self.bacformer(
            protein_embeddings=None,
            labels=labels,
            special_tokens_mask=special_tokens_mask,
            token_type_ids=token_type_ids,
            property_ids=property_ids,
            return_attn_weights=return_attn_weights,
            return_dict=return_dict,
            is_causal=True,
        )
        last_hidden_state = outputs[0]
        prediction_scores = self.gm_head(last_hidden_state)

        loss = None
        if labels is not None:
            if property_ids is not None:
                batch_size = labels.shape[0]
                labels = torch.cat(
                    [
                        torch.ones(batch_size, 1, dtype=torch.long).to(labels.device),  # account for the property token
                        labels,
                    ],
                    dim=1,
                )  # ignore index
            labels = labels.to(prediction_scores.device)

            shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous().view(-1, prediction_scores.shape[-1])
            labels = labels[:, 1:].contiguous().view(-1)
            loss = cross_entropy(shifted_prediction_scores, labels)

        if not return_dict:
            return (
                loss,
                prediction_scores,
            ) + outputs

        return BacformerModelOutput(
            loss=loss,
            logits=prediction_scores,
            last_hidden_state=outputs.last_hidden_state,
            attentions=outputs.attentions,
        )

    def generate(
        self,
        protein_family_ids: torch.LongTensor,
        special_tokens_mask: torch.LongTensor = None,
        token_type_ids: torch.LongTensor = None,
        max_length: int = 6000,
        end_token_id: int = 50000,
        do_sample: bool = False,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 1.0,
        property_ids: torch.LongTensor = None,
        return_last_hidden_states: bool = False,
    ):
        """
        Generate a sequence of tokens autoregressively from a given prompt.

        Args:
            protein_family_ids (torch.LongTensor): Tensor of shape (batch, seq_len) with token indices.
            max_length (int): Maximum length of the generated sequence (prompt + newly generated).
            end_token_id (int, optional): Token ID signifying end-of-sequence (END).
                                          If encountered, generation stops.
            do_sample (bool): Whether to sample from the probability distribution (True)
                              or use greedy decoding (False).
            top_k (int): If >0, use top-k filtering in sampling mode.
            top_p (float): If <1.0, use nucleus (top-p) filtering in sampling mode.
            temperature (float): Softmax temperature for scaling logits.
                                 Higher => more random, lower => more deterministic.
            return_last_hidden_states (bool): If True, return final hidden states as well.

        Returns
        -------
            torch.LongTensor: The generated token sequence of shape (batch, final_seq_len).
            (Optional) torch.FloatTensor: Final hidden states of shape (batch, final_seq_len, hidden_dim)
                                          if `return_hidden_states=True`.
        """
        # Default END token
        if end_token_id is None:
            end_token_id = getattr(self, "end_token_id", None)

        # Switch to eval mode and move input to correct device
        self.eval()
        device = next(self.parameters()).device
        protein_family_ids = protein_family_ids.to(device)

        # create a special tokens mask if not provided
        if special_tokens_mask is None:
            # add a cls token at the beginning
            protein_family_ids = torch.cat(
                [torch.tensor([[-100]]).to(device), protein_family_ids],
                dim=1,
            )
            special_tokens_mask = [self.cls_token_id] + [self.config.prot_emb_token_id] * (
                protein_family_ids.shape[1] - 1
            )
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.long).to(device)

        # create a token type mask if not provided
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(protein_family_ids)

        # Prepare the initial sequence and define max new tokens
        generated = protein_family_ids.clone()
        batch_size, prompt_length = generated.shape
        max_new_tokens = max_length - prompt_length
        if max_new_tokens <= 0:
            max_new_tokens = 0

        # Disable gradient calculations for generation
        with torch.no_grad():
            for _step in range(max_new_tokens):
                # Forward pass
                logits = self.forward(
                    labels=generated,
                    special_tokens_mask=special_tokens_mask,
                    # assume it's all on one chromosome
                    token_type_ids=token_type_ids,
                    property_ids=property_ids,
                    return_dict=True,
                ).logits
                # Focus on the last token's logits
                next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)

                # Apply temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature

                # Sampling or greedy?
                if do_sample:
                    # Top-k filter
                    next_token_logits = top_k_filtering(next_token_logits, top_k=top_k)
                    # Top-p filter
                    next_token_logits = top_p_filtering(next_token_logits, top_p=top_p)

                    probs = softmax(next_token_logits, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # Append predicted token
                generated = torch.cat([generated, next_token_id], dim=1)
                special_tokens_mask = torch.cat(
                    [special_tokens_mask, torch.tensor([[self.config.prot_emb_token_id]]).to(generated.device)], dim=1
                )
                last_token_type_id = token_type_ids[:, -1].unsqueeze(1)
                token_type_ids = torch.cat([token_type_ids, last_token_type_id], dim=1)

                # Check for END in all sequences
                if end_token_id is not None:
                    if (next_token_id.squeeze(1) == end_token_id).all():
                        # If every sequence ended, break early
                        break

        if not return_last_hidden_states:
            return generated

        # Optionally compute final hidden states
        if return_last_hidden_states:
            last_hidden_state = self.forward(
                labels=generated,
                special_tokens_mask=special_tokens_mask,
                token_type_ids=token_type_ids,
                return_dict=True,
            ).last_hidden_state

        return generated, last_hidden_state


class BacformerForMaskedGMWithContrastiveLoss(BacformerPreTrainedModel):
    """Bacformer model for masked genomic modeling with contrastive loss.

    The contrastive loss tries to minimise the distance between the protein embeddings used as input
    and the last hidden state of the model. The model takes as input a sequence of protein embeddings.
    """

    _tied_weights_keys = ["gm_head.decoder.weight"]

    def __init__(self, config: BacformerConfig):
        super().__init__(config)
        self.config = config

        self.bacformer = BacformerModel(config, add_pooling_layer=False)
        self.gm_head = BacformerGMHead(config)

        # Initialize weights
        self.init_weights()

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

        # to speed up the forward pass, let's only consider the masked tokens

        loss = None
        if labels is not None:
            # contrastive loss
            contrastive_loss = compute_contrastive_loss(protein_embeddings, last_hidden_state, special_tokens_mask)
            # to speed up the forward pass, let's only consider the masked tokens
            last_hidden_state = last_hidden_state[labels != -100]
            prediction_scores = self.gm_head(last_hidden_state)
            labels = labels.to(prediction_scores.device)

            # only considering the masked tokens
            labels = labels[labels != -100]
            masked_loss = cross_entropy(prediction_scores, labels)
            loss = masked_loss + self.config.alpha_contrastive_loss * contrastive_loss
        else:
            prediction_scores = self.gm_head(last_hidden_state)

        if not return_dict:
            return (
                loss,
                prediction_scores,
            ) + outputs

        return BacformerModelOutput(
            loss=loss,
            logits=prediction_scores,
            last_hidden_state=outputs.last_hidden_state,
            attentions=outputs.attentions,
        )
