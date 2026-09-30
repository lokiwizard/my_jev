"""Open-Jev 文本 backbone + LoRA + 单标量头；候选之间不共享训练注意力。"""

from __future__ import annotations

import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoModelForImageTextToText, AutoTokenizer

from .data import render_prompts


class DecisionModel(nn.Module):
    def __init__(self, config: dict, device: str):
        super().__init__()
        self.model_config = dict(config)
        self.sort_candidates = config["sort_candidates"]
        self.device_name, self.max_length = device, config["max_length"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["id"],
            revision=config["revision"],
            local_files_only=True,
        )
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = getattr(torch, config["dtype"])
        # Qwen3.5 的 HF checkpoint 包含多模态外壳；取出文本 backbone 后丢弃外壳。
        full = AutoModelForImageTextToText.from_pretrained(
            config["id"],
            revision=config["revision"],
            local_files_only=True,
            dtype=dtype,
            attn_implementation="sdpa",
            device_map={"": device},
            # Transformers 5.10.2 的 PEFT 自动探测不继承外层 Hub 参数，需再次声明。
            adapter_kwargs={"revision": config["revision"], "local_files_only": True},
        )
        yes = self.tokenizer.encode("Yes", add_special_tokens=False)
        no = self.tokenizer.encode("No", add_special_tokens=False)
        if len(yes) != 1 or len(no) != 1:
            raise ValueError("评分头初始化要求 Yes / No 分别是单 token")
        readout = full.get_output_embeddings().weight
        initial = (readout[yes[0]] - readout[no[0]]).detach().float().clone()
        self.backbone = full.model.language_model
        hidden_size = self.backbone.config.hidden_size
        del full, readout
        self.backbone.requires_grad_(False)
        self.backbone.config.use_cache = False
        self.head = nn.Linear(hidden_size, 1, dtype=torch.float32, device=device)
        with torch.no_grad():
            self.head.weight.copy_(initial.unsqueeze(0))
            self.head.bias.zero_()
        if config["lora_rank"]:
            names = {name.rsplit(".", 1)[-1] for name, _ in self.backbone.named_modules()}
            intended = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "out_proj"]
            targets = [name for name in intended if name in names]
            if not targets:
                raise ValueError("没有找到预期的 LoRA 投影模块")
            self.backbone = get_peft_model(
                self.backbone,
                LoraConfig(
                    r=config["lora_rank"],
                    lora_alpha=config["lora_alpha"],
                    lora_dropout=config["lora_dropout"],
                    target_modules=targets,
                    bias="none",
                ),
            )
        if config["gradient_checkpointing"]:
            self.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
            self.backbone.enable_input_require_grads()

    def forward(self, records: list[dict], candidate_batch_size: int = 4) -> list[torch.Tensor]:
        """records [B] → 每个问题的 logits [K]；保留全部候选的梯度再计算损失。

        每个 chunk 的 input_ids [N,L]、hidden [N,L,D]；right padding 时读取 lengths-1。
        Noul 只有一条评分序列，但返回 [0,s] 两个 logits。训练不使用 KV cache。
        """
        if not records or candidate_batch_size < 1:
            raise ValueError("records 不能为空且 candidate_batch_size 必须为正数")
        prompts, counts = [], []
        for record in records:
            entries = render_prompts(record, self.tokenizer)
            counts.append(len(entries))
            prompts.extend(entries)
        # 可选：仅在本微批内把长度相近的独立序列排在一起，减少 padding。
        # 不拼接上下文、不共享注意力；评分完成后恢复原决策/候选顺序。
        prepared, order = None, None
        if self.sort_candidates:
            prepared = self.tokenizer(prompts, padding=False, truncation=False)["input_ids"]
            if max(map(len, prepared)) > self.max_length:
                raise ValueError(f"输入长度超过 {self.max_length}，禁止静默截断")
            order = sorted(range(len(prepared)), key=lambda index: len(prepared[index]))
        scores = []
        for start in range(0, len(prompts), candidate_batch_size):
            if prepared is None:
                encoded = self.tokenizer(
                    prompts[start : start + candidate_batch_size],
                    padding=True,
                    truncation=False,
                    return_tensors="pt",
                ).to(self.device_name)
            else:
                tokens = [prepared[index] for index in order[start : start + candidate_batch_size]]
                length = max(map(len, tokens))
                encoded = {
                    "input_ids": torch.tensor(
                        [
                            row + [self.tokenizer.pad_token_id] * (length - len(row))
                            for row in tokens
                        ],
                        device=self.device_name,
                    ),
                    "attention_mask": torch.tensor(
                        [[1] * len(row) + [0] * (length - len(row)) for row in tokens],
                        device=self.device_name,
                    ),
                }
            lengths = encoded["attention_mask"].sum(-1)
            if prepared is None and lengths.max().item() > self.max_length:
                raise ValueError(f"输入长度超过 {self.max_length}，禁止静默截断")
            outputs = self.backbone(**encoded, use_cache=False, return_dict=True)
            hidden = outputs.last_hidden_state
            rows = torch.arange(hidden.shape[0], device=hidden.device)
            selected = hidden[rows, lengths - 1]
            scores.append(self.head(selected.float()).squeeze(-1))
        scores = torch.cat(scores)
        if order is not None:
            inverse = sorted(range(len(order)), key=lambda index: order[index])
            scores = scores[torch.tensor(inverse, device=scores.device)]
        logits, offset = [], 0
        for record, count in zip(records, counts):
            values = scores[offset : offset + count]
            if record["kind"] == "noul":
                values = torch.stack([torch.zeros_like(values[0]), values[0]])
            logits.append(values)
            offset += count
        return logits

    def trainable_state(self) -> dict:
        """只保存 LoRA/head；冻结 backbone 由固定 revision 重建，不复制基础权重。"""
        return {
            name: value.detach().cpu().clone()
            for name, value in self.named_parameters()
            if value.requires_grad
        }

    def load_trainable_state(self, saved: dict) -> None:
        current = {name: value for name, value in self.named_parameters() if value.requires_grad}
        if set(current) != set(saved):
            raise ValueError("checkpoint 的 LoRA/head 参数名与当前模型不一致")
        with torch.no_grad():
            for name, value in current.items():
                source = saved[name]
                if value.shape != source.shape or value.dtype != source.dtype:
                    raise ValueError(f"checkpoint 参数 shape/dtype 不符: {name}")
                if not torch.isfinite(source).all():
                    raise ValueError(f"checkpoint 存在非有限参数: {name}")
                value.copy_(source.to(value.device))

    def score_cached(self, records: list[dict], candidate_batch_size: int = 4):
        """显式可选的实验性推理缓存；复用 upstream Qwen 混合状态分支实现。"""
        from ._upstream.prefix_cache import score_cached

        return score_cached(self, records, batch_size=candidate_batch_size)


def build_model(config: dict) -> DecisionModel:
    from .checkpoint import configure_runtime

    configure_runtime(config["runtime"])
    device = config["runtime"]["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("配置要求 CUDA，但当前环境无法使用 GPU")
    return DecisionModel(config["model"], device)
