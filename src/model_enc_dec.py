"""
    基于Hi-VT5,采用encoder-decoder架构,而且encoder和decoder均才用qwen-vl的模型结构
"""
import logging
from typing import Optional, Tuple, List

import torch
from torch import nn
from transformers import (
    PreTrainedModel,
    GenerationConfig,
)
from peft.peft_model import PeftModel
from peft import get_peft_model, LoraConfig, TaskType

from qwen_vl import QWenLMHeadModel, QWenConfig, QWenTokenizer
from utils import seed_everything

ONE_BATCH = 0
ONE_SEQ = 1
TWO_SEQ = 2

IGNORE_INDEX = -100

logger = logging.getLogger(__name__)


class MPModelConfig(QWenConfig):
    def __init__(
        self,
        classification_head_hidden_size: int = 4096,
        num_pages: int = 20,
        padding=True,
        padding_side="right",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.classification_head_hidden_size = classification_head_hidden_size
        self.num_pages = num_pages
        self.padding = padding
        self.padding_side = padding_side


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
        return self.mlp[0].weight.device


class MPModel(PreTrainedModel):
    def __init__(self, config: MPModelConfig, qwen_vl: nn.Module):
        super().__init__(config)
        self.head = ClassificationHeadForPageIndex(config)
        self.config = config
        self.qwen_vl = qwen_vl
        assert self.qwen_vl.generation_config is not None

    def get_cast_dtype(self):
        if self.config.bf16:
            return torch.bfloat16
        elif self.config.fp16:
            return torch.float16
        elif self.config.fp32:
            return torch.float32
        else:
            raise ValueError("no supported dtype")

    def get_cast_device(self):
        return self.head.get_cast_device()

    def get_cast_device(self):
        pass

    def get_last_hidden_states(
        self,
        output_sequences: torch.LongTensor,
        hidden_states: Tuple[Tuple[torch.Tensor]],
    ) -> List[torch.Tensor]:
        """
        output_sequences: [cnt, seq_len]
            需要提前删除前置prompt对应的id
        hidden_statts: tuple, len is seq_len,
        elements is tuple , len is transformer layer num,
        inner elements is torch.Tensor, size is [cnt, seq_len, hidden_size]
        for prompt , seq_len is prompt_len
        for other new predict token, seq_len is 1

        return last_hidden_states: List[torch.Tensor], len is cnt
        elments is torch.Tensor, size is [new_seq_len, hidden_size]
        """
        cnt = output_sequences.size(0)
        all_last_hidden_states = [[] for _ in range(cnt)]
        eos_id = self.config.pad_token_id
        # loop bsz
        for cnt_idx in range(cnt):
            seq_end_idx = (
                output_sequences[cnt_idx, :].view(-1).cpu().tolist().index(eos_id)
            )
            # loop seq_len
            for seq_idx, token_hidden_states in enumerate(hidden_states):
                # ignore eos token
                if seq_idx >= seq_end_idx:
                    break
                last_hidden_states = token_hidden_states[-1][cnt_idx]
                if seq_idx == 0 and last_hidden_states.size(0) != 1:
                    last_hidden_states = last_hidden_states[-1:, :]
                all_last_hidden_states[cnt_idx].append(last_hidden_states.view(-1))
        for idx in range(cnt):
            all_last_hidden_states[idx] = torch.stack(
                all_last_hidden_states[idx], dim=0
            )
        return all_last_hidden_states

    def encode(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor,
        page_mask: torch.LongTensor,
        encode_mode: int = TWO_SEQ,
    ):
        """
        input_ids: [bsz, num_page, seq_len]
        attention_mask: [bsz, num_page, seq_len]
        page_mask: [bsz, num_page]
        page_nums: [bsz]
        """
        assert encode_mode >= 0
        page_nums = page_mask.sum(dim=-1)
        encoder_outputs = []
        bsz = input_ids.size(0)
        for idx in range(bsz):
            # 下面这段if没有测试过,因此也不知道是否可以正常工作
            if encode_mode == ONE_BATCH:
                one_input_ids = (
                    input_ids[idx, : page_nums[idx], :],
                )  # [cur_num_page, seq_len]
                one_attention_mask = (
                    attention_mask[idx, : page_nums[idx], :],
                )  # [cur_num_page, seq_len]
                # [cur_num_page,seq_len,hidden_size]
                generate_outputs = self.qwen_vl.generate(
                    inputs=part_input_ids,
                    attention_mask=part_attention_mask,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                    max_new_tokens=128,
                    generation_config=self.qwen_vl.generation_config,
                )
                output_sequences = generate_outputs.sequences[
                    :, self.config.prompt_len :
                ]
                hidden_states = generate_outputs.hidden_states
                # list, len is cur_num_page, elem is torch.FloatTensor, size is [seq_len, hidden_size]
                last_hidden_states = self.get_last_hidden_states(
                    output_sequences, hidden_states
                )
                encoder_outputs.append(last_hidden_states)
            else:
                cur_data_outputs = []
                # [encode_mode, seq_len, hidden_size]
                step = min(encode_mode, page_nums[idx])
                for st in range(0, int(page_nums[idx]), step):
                    ed = min(st + step, page_nums[idx])
                    # [encode_mode, seq_len, hidden_size]
                    part_input_ids = input_ids[idx, st:ed, :]
                    part_attention_mask = attention_mask[idx, st:ed, :]
                    generate_outputs = self.qwen_vl.generate(
                        inputs=part_input_ids,
                        attention_mask=part_attention_mask,
                        return_dict_in_generate=True,
                        output_hidden_states=True,
                        max_new_tokens=128,
                        generation_config=self.qwen_vl.generation_config,
                    )
                    output_sequences = generate_outputs.sequences[
                        :, self.config.prompt_len :
                    ]
                    hidden_states = generate_outputs.hidden_states
                    cur_last_hidden_states = self.get_last_hidden_states(
                        output_sequences, hidden_states
                    )
                    cur_data_outputs.extend(cur_last_hidden_states)
                encoder_outputs.append(cur_data_outputs)

        return encoder_outputs

    def _prepare_decoder_inputs(
        self,
        encoder_outputs: List[torch.FloatTensor],
        labels: Optional[List[List[int]]] = None,
        padding=False,
        max_seq_len: int = 512,
        padding_side="right",
    ):
        """
        Args:
        encoder_outputs:
            List[List[torch.FloatTensor]]
            1. outer List is batch, so len(encoder_outputs) is batch_size
            2. inner List is page, so len(encoder_outputs[idx]) is cur_num_page, every question is different
            3. inner List elem is torch.FloatTensor, size is [cur_seq_len, hidden_size],
            every (question,page) pair have different shape in 'cur_seq_len'
        labels: List[List[int]]
            1. outer List is batch, so len(labels) is batch_size
            2. inner List is answer input_ids, so len(labels[idx]) = answer_seq_len, every answer is different
        padding: whether or not paddin inputs
        padding_side: padding direction
        max_seq_len: padding to this length
        return:
            decoder_inputs: [bsz, max_seq_len, hidden_size]
            decoder_attention_mask: [bsz, max_seq_len]
            decoder_labels: [bsz, max_seq_len] | None
        """
        have_labels = labels is not None
        bsz = len(encoder_outputs)
        decoder_emb_list = [[] for _ in range(bsz)]  # elem is list
        decoder_inputs = []  # return
        decoder_labels = [] if have_labels else None  # return
        valid_seq_len = [0 for _ in range(bsz)]  # calculate mask

        # prompt embedding
        for idx in range(bsz):
            one_encoder_outputs = encoder_outputs[idx]
            encoder_prompt_embedding = torch.cat(
                one_encoder_outputs, dim=0
            )  # [seq1+seq2+... ,hidden_size]
            valid_seq_len[idx] += encoder_prompt_embedding.size(0)
            decoder_emb_list[idx].append(encoder_prompt_embedding)
        # answer embedding
        device = encoder_prompt_embedding.device
        if have_labels:
            for idx in range(bsz):
                # [answer_seq_len, hidden_size]
                cur_labels = torch.tensor(
                    labels[idx],
                    dtype=torch.long,
                    device=device,
                )
                answers_embedding = self.qwen_vl.transformer.wte(cur_labels)
                answer_len = answers_embedding.size(0)
                decoder_label = ([IGNORE_INDEX] * valid_seq_len[idx]) + labels[idx]
                decoder_labels.append(decoder_label)
                decoder_emb_list[idx].append(answers_embedding)
                valid_seq_len[idx] += answer_len
        # padding embedding
        if padding:
            assert self.config.pad_token_id is not None
            padding_embedding = self.qwen_vl.transformer.wte(
                torch.tensor(
                    [self.config.pad_token_id], dtype=torch.long, device=device
                )
            ).view(1, -1)

            def padding_to_left(emb_list, padding_emb, label=None):
                emb_list = [padding_emb].extend(emb_list)
                padding_len = padding_emb.size(0)
                if label is not None:
                    label = ([IGNORE_INDEX] * padding_len) + label
                return emb_list, label

            def padding_to_right(emb_list, padding_emb, label=None):
                emb_list.append(padding_emb)
                padding_len = padding_emb.size(0)
                if label is not None:
                    label = label + ([IGNORE_INDEX] * padding_len)
                return emb_list, label

            for idx in range(bsz):
                padding_len = max_seq_len - valid_seq_len[idx]
                if padding_len > 0:
                    all_padding_embedding = torch.repeat_interleave(
                        padding_embedding, padding_len, dim=0
                    )
                    emb_list = decoder_emb_list[idx]
                    no_padding_labels = decoder_labels[idx] if have_labels else None
                    if padding_side == "right":
                        decoder_emb_list[idx], o2 = padding_to_right(
                            emb_list, all_padding_embedding, no_padding_labels
                        )
                        if o2:
                            decoder_labels[idx] = o2
                    elif padding_side == "left":
                        decoder_emb_list[idx], o2 = padding_to_left(
                            emb_list, all_padding_embedding, no_padding_labels
                        )
                        if o2:
                            decoder_labels[idx] = o2
                    else:
                        raise ValueError("no supported padding side")
                else:
                    # padding < 0, will handle in next loop
                    pass
        # cat
        for idx in range(bsz):
            padding_len = max_seq_len - valid_seq_len[idx]
            # [seq1+seq2+...+answer_seq_len+padding_seq_len, hidden_size]
            decoder_input = torch.cat(decoder_emb_list[idx], dim=0)
            if padding_len < 0:
                decoder_input = decoder_input[:max_seq_len, :]
                valid_seq_len[idx] = max_seq_len
            decoder_inputs.append(decoder_input)

        # decoder_inputs: [bsz, max_seq_len, hidden_size]
        decoder_inputs = torch.stack(decoder_inputs, dim=0)
        # decoder_labels: [bsz, max_seq_len]
        if have_labels:
            decoder_labels = torch.tensor(
                decoder_labels, dtype=torch.long, device=device
            )
        decoder_attention_mask = self._prepare_decoder_input_attention_mask(
            max_seq_len, valid_seq_len, device=device
        )
        return decoder_inputs, decoder_attention_mask, decoder_labels

    def _prepare_decoder_input_attention_mask(
        self, max_seq_len: int, valid_seq_len: List[int], device: str
    ) -> torch.LongTensor:
        mask = torch.arange(max_seq_len, device=device)  # [max_seq_len]
        mask = mask.unsqueeze(0)  # [1, max_seq_len]
        vaild_seq_len = torch.tensor(valid_seq_len, device=device).view(
            -1, 1
        )  # [bsz,1]
        mask = mask < vaild_seq_len  # [bsz, max_seq_len]

        return mask.long()

    def decode(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return self.qwen_vl.forward(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor,
        page_mask: torch.LongTensor,
        labels: Optional[List[List[int]]] = None,
        answer_page_index: Optional[torch.LongTensor] = None,
    ):
        """
        Args:
            input_ids: [bsz, max_num_page, seq_len]
            page_mask: [bsz, max_num_page]
            attention_mask: [bsz, max_num_page, seq_len]
            labels: List[List[int]]
            answer_page_index: [bsz]
        """
        # step1. encode
        encode_outputs = self.encode(
            input_ids=input_ids,
            attention_mask=attention_mask,
            page_mask=page_mask,
            encode_mode=self.config.encode_mode,
        )
        # step2. prepare decode inputs
        (
            decoder_input_embeds,
            decoder_attention_mask,
            decoder_labels,
        ) = self._prepare_decoder_inputs(
            encoder_outputs=encode_outputs,
            labels=labels,
            max_seq_len=self.config.model_max_length,
            padding=True,
        )

        decoder_outputs = self.decode(
            input_ids=None,
            inputs_embeds=decoder_input_embeds,
            attention_mask=decoder_attention_mask,
            labels=decoder_labels,
            return_dict=False,
        )
        if labels is not None:
            decoder_loss, lm_logits = decoder_outputs[0], decoder_outputs[1]
        else:
            lm_logits = decoder_outputs[0]

        # step4. classification head

        # classification_loss is None if answer_page_index is None
        device = decoder_input_embeds.device
        answer_page_index = answer_page_index.to(device)
        classification_loss, logits = self.head(decoder_input_embeds, answer_page_index)
        if labels is not None:
            return decoder_loss + classification_loss, lm_logits, logits
        else:
            return lm_logits, logits


def load_qwen_vl_model_tokenizer(
    model_name_or_path: str, config: MPModelConfig, device_map: str = "cpu"
) -> Tuple[QWenLMHeadModel, QWenTokenizer]:
    model = load_qwen_vl_model(model_name_or_path, config, device_map)
    tokenizer = load_qwen_vl_tokenizer(model_name_or_path)
    return model, tokenizer


def load_qwen_vl_model(
    model_name_or_path: str, config: MPModelConfig, device_map: str = "auto"
):
    # logger.info(
    #     f"load_qwen_vl_model params: {model_name_or_path}, \n{config},\n {device_map}"
    # )
    qwen_vl = QWenLMHeadModel.from_pretrained(
        model_name_or_path, config=config, device_map=device_map
    )
    qwen_vl.generation_config = GenerationConfig.from_pretrained(model_name_or_path)
    return qwen_vl


def load_qwen_vl_tokenizer(
    model_name_or_path: str, model_max_length: int = 1024, padding_side="left"
):
    tokenizer = QWenTokenizer.from_pretrained(
        model_name_or_path, model_max_length=model_max_length, padding_side=padding_side
    )
    tokenizer.pad_token = "<|endoftext|>"
    tokenizer.eos_token = "<|endoftext|>"
    return tokenizer


def load_config(
    model_name_or_path: str,
    pad_token_id: int,
    model_max_length: int = 1024,
):
    config = MPModelConfig.from_pretrained(
        model_name_or_path,
        num_pages=40,
    )
    config.bf16 = True
    config.pad_token_id = pad_token_id
    config.model_max_length = model_max_length
    config.prompt_len = model_max_length
    config.encode_mode = ONE_SEQ
    return config


def load_qwen_vl_model_lora_from_scratch(
    model_name_or_path: str,
    config: MPModelConfig,
    device: str = "cuda:0",
    lora_r: int = 4,
) -> PeftModel:
    model = load_qwen_vl_model(model_name_or_path, config, device)
    target_modules = ["attn.c_attn", "attn.c_proj"]
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=target_modules,
    )
    lora_model = get_peft_model(model, lora_config, adapter_name="mpdoc_qwen-vl_lora")
    lora_model.print_trainable_parameters()
    return lora_model


if __name__ == "__main__":
    from dataset import (
        MPDocVQADatasetForEncoderDecoderModel,
        CollatorForEncoderDecoderModel,
    )

    seed_everything(42)

    train_dataset = MPDocVQADatasetForEncoderDecoderModel(
        json_path="../data/MPDocVQA/train.json",
        image_dir="../data/MPDocVQA/images",
        split="train",
    )
    model_path = "../pretrain-model/QWen-VL/"
    tokenizer = load_qwen_vl_tokenizer(
        model_path, model_max_length=512, padding_side="left"
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=CollatorForEncoderDecoderModel(tokenizer),
    )
    config = load_config(
        model_path, pad_token_id=tokenizer.pad_token_id, model_max_length=1024
    )
    device = "cuda:0"
    qwen_vl = load_qwen_vl_model_lora_from_scratch(
        model_path, config, device=device, lora_r=4
    )
    model = MPModel(config=config, qwen_vl=qwen_vl)
    # for name, param in model.named_parameters():
    #     param.requires_grad = False
    model.to(device).to(torch.bfloat16)

    print("model device:", model.device)
    print("moedl.qwen_vl device:", model.qwen_vl.device)
    print("model.head device:", model.head.get_cast_device())

    for idx, batch in enumerate(train_dataloader):
        if idx >= 10:
            break
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)

        loss, lm_logits, idx_logits = model(**batch)
        print(f"batch: {idx}\tloss:{loss.data:.4f}")
