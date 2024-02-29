"""
    qwen-vl模型+分类头
"""
from typing import Optional, Tuple, List
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


class MPPretrainedModel(PreTrainedModel):
    config_class = MPModelConfig
    base_model_prefix = "qwen_vl"

    def __init__(self, config: MPModelConfig):
        super().__init__(config)
        self.config = config

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, MPModel):
            module.gradient_checkpointing = value

class MPModel(MPPretrainedModel):

    def __init__(self, config: MPModelConfig, qwen_vl: QWenLMHeadModel):
        "qwen_vl is PretrainedModel or PeftModel"
        super().__init__(config)
        self.config = config
        self.qwen_vl = qwen_vl
        self.head = ClassificationHeadForPageIndex(config)
        self.gradient_checkpointing = False

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
        if self.gradient_checkpointing:
            head_loss, head_logits = torch.utils.checkpoint(
                self.head.__call__, last_hidden_state, page_idx_labels
            )
        else:
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

    def get_nb_trainable_parameters(self):
        r"""
        Returns the number of trainable parameters and number of all parameters in the model.
        """
        # note: same as PeftModel.get_nb_trainable_parameters
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()

            # Due to the design of 4bit linear layers from bitsandbytes
            # one needs to multiply the number of parameters by 2 to get
            # the correct number of parameters
            if param.__class__.__name__ == "Params4bit":
                num_params = num_params * 2

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param

    def print_trainable_parameters(self):
        r"""
        Prints the number of trainable parameters and number of all parameters in the model.
        """
        # note: same as PeftModel.print_trainable_parameters
        trainable_params, all_param = self.get_nb_trainable_parameters()
        print(
            f"{self.__class__.__name__}: trainable params: {trainable_params:,d} || "
            f"all params: {all_param:,d} || "
            f"trainable%: {100 * trainable_params / all_param:.4f}"
        )


def load_lora_qwen_vl_model(
    qwen_vl: QWenLMHeadModel,
    r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: List[str],
    lora_bias: str = "none",
    adapter_name: str = "default",
):
    # target_modules = ["c_attn", "c_proj"]
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules,
        bias=lora_bias,
    )
    lora_model = get_peft_model(qwen_vl, lora_config, adapter_name=adapter_name)
    lora_model.print_trainable_parameters()
    return lora_model
