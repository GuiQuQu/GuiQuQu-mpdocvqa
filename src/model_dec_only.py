"""
    qwen-vl模型+分类头
"""
from typing import Optional, Tuple
from dataclasses import dataclass

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.utils import ModelOutput
from peft import get_peft_model, LoraConfig, LoraModel, TaskType

from qwen_vl import QWenConfig, QWenLMHeadModel


class MPModelConfig(QWenConfig):
    def __init__(
        self, classification_head_hidden_size: int = 4096, num_pages: int = 20, **kwargs
    ):
        super().__init__(**kwargs)
        self.classification_head_hidden_size = classification_head_hidden_size
        self.num_pages = num_pages


class ClassificationHeadForPageIndex(nn.Module):
    def __init__(self, config: MPModelConfig):
        super().__init__()
        hidden_size = config.classification_head_hidden_size
        num_classes = config.num_pages
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_classes),
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor):
        """
        inputs: [batch_size, seq_length, hidden_size]
        labels: [batch_size] is answer_page_index
        """
        inputs = torch.mean(inputs, dim=-2)
        logits = self.mlp(inputs)
        logits = self.softmax(logits)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return loss, logits

    def get_cast_device(self):
        return next(self.parameters()).device


@dataclass
class MPModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    lm_logits: Optional[torch.FloatTensor] = None
    classification_logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class MPModel(PreTrainedModel):
    def __init__(self, config: MPModelConfig, qwen_vl: QWenLMHeadModel):
        super().__init__(config)
        self.qwen_vl = qwen_vl
        # if not hasattr(self.qwen_vl, "generation_config"):
        #     raise ValueError("qwen_vl need has 'generation_config' attr to generate")
        self.head = ClassificationHeadForPageIndex(config)

    def get_cast_dtype(self):
        return next(self.parameters()).dtype

    def get_cast_device(self):
        return next(self.parameters()).device

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        page_idx_labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = True
        qwen_vl_outputs = self.qwen_vl(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        lm_logits = qwen_vl_outputs.logits
        last_hidden_state = qwen_vl_outputs.hidden_states[-1]
        qwen_vl_loss = None
        if labels is not None:
            qwen_vl_loss = qwen_vl_outputs.loss

        # classification head
        head_loss, head_logits = self.head(last_hidden_state, page_idx_labels)
        loss = None
        if head_loss is not None and qwen_vl_loss is not None:
            loss = qwen_vl_loss + head_loss

        if not return_dict:
            outputs = (
                lm_logits,
                head_logits,
                qwen_vl_outputs.hidden_states,
                qwen_vl_outputs.attentions,
            )
            return ((loss,) + outputs) if loss is not None else outputs

        return MPModelOutput(
            loss=loss,
            lm_logits=lm_logits,
            classification_logits=head_logits,
            hidden_states=qwen_vl_outputs.hidden_states,
            attentions=qwen_vl_outputs.attentions,
        )

    def generate(self, **kwargs):
        self.qwen_vl.generate(**kwargs)


def load_lora_qwen_vl_model(
    qwen_vl: QWenLMHeadModel,
    r: int,
    lora_alpha: int,
    lora_dropout: float,
    adapter_name: str = "default",
):
    target_modules = ["c_attn", "c_proj"]
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
    )
    lora_model = get_peft_model(qwen_vl, lora_config, adapter_name=adapter_name)
    lora_model.print_trainable_parameters()
    return lora_model
