import os
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Sampler
from transformers import Trainer
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.trainer import (
    ALL_LAYERNORM_LAYERS,
    get_parameter_names,
    has_length,
    is_sagemaker_mp_enabled,
    logger,
    _is_peft_model
)
from vita.constants import MCCD, AUDIO_TOKEN_INDEX
from vita.conversation import conv_mixtral_two
import pdb
def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

def get_peft_state_maybe_zero_3(named_params, bias):
    # print("---------------------bias: ", bias)
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError

    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {
        k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)
    }
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [
        mm_indices[i]
        for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)
    ]
    lang_shuffle = [
        lang_indices[i]
        for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)
    ]
    megabatch_size = world_size * batch_size
    mm_megabatches = [
        mm_shuffle[i: i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)
    ]
    lang_megabatches = [
        lang_shuffle[i: i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)
    ]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [
        indices[i: i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)
    ]
    megabatches = [
        sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches
    ]
    megabatches = [
        split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches
    ]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
            self,
            batch_size: int,
            world_size: int,
            lengths: Optional[List[int]] = None,
            generator=None,
            group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(
                self.lengths, self.batch_size, self.world_size, generator=self.generator
            )
        else:
            indices = get_length_grouped_indices(
                self.lengths, self.batch_size, self.world_size, generator=self.generator
            )
        return iter(indices)


class VITATrainer(Trainer):
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    # add this function to implement MCCD (Jie).
    def compute_loss(self, model, inputs, return_outputs=False):
        if MCCD["flag"]:
            print("=" * 80)
            print("Please note that MCCD is true. It is used to perform training bias eliminating.")
            print("You can set its value into the TrainingArguments within train.py")

            if self.label_smoother is not None and "labels" in inputs:
                labels = inputs.pop("labels")
            else:
                labels = None
            outputs = model(**inputs)
            # Save past state if it exists
            # TODO: this needs to be fixed and made cleaner later.
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index]

            # Obtain the uni-modal logits including audio, video, and question logits
            prompt_lang_only = conv_mixtral_two.system[3]
            prompt_audio_only = conv_mixtral_two.system[4]
            prompt_video_only = conv_mixtral_two.system[5]

            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model = unwrapped_model.base_model.model
            else:
                model = unwrapped_model

            input_emb_lang, input_attn_mask = self.get_lang_inputs(prompt_lang_only, inputs, model)
            outputs_lang = model(inputs_embeds=input_emb_lang, attention_mask=input_attn_mask)

            input_emb_audio, input_attn_mask = self.get_audio_inputs(prompt_audio_only, inputs, model)
            outputs_audio = model(inputs_embeds=input_emb_audio, attention_mask=input_attn_mask)

            batch_size = input_emb_audio.shape[0]
            input_emb_video, input_attn_mask = self.get_video_inputs(prompt_video_only, inputs, model, batch_size)
            outputs_video = model(inputs_embeds=input_emb_video, attention_mask=input_attn_mask)

            loss_mccd = self.mccd(
                logit_qs=outputs_lang["logits"],
                logit_audio=outputs_audio["logits"],
                logit_video=outputs_video["logits"],
                logit_fusion=outputs["logits"]
            )
            if labels is not None:
                unwrapped_model = self.accelerator.unwrap_model(model)
                if _is_peft_model(unwrapped_model):
                    model_name = unwrapped_model.base_model.model._get_name()
                else:
                    model_name = unwrapped_model._get_name()
                if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                    loss = self.label_smoother(outputs, labels, shift_labels=True)
                else:
                    loss = self.label_smoother(outputs, labels)
            else:
                if isinstance(outputs, dict) and "loss" not in outputs:
                    raise ValueError(
                        "The model did not return a loss from the inputs, only the following keys: "
                        f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                    )
                # We don't use .loss here since the model may return tuples instead of ModelOutput.
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

            print("loss_mccd: ", loss_mccd.item())
            print("loss_qa: ", loss.item())
            return (loss+loss_mccd, outputs) if return_outputs else loss+loss_mccd
        else:
            return super(VITATrainer, self).compute_loss(model, inputs)

    def get_lang_inputs(self, prompt_lang_only, inputs, model):
        input_ids_lang = self.tokenizer(
            prompt_lang_only,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )

        # with torch.no_grad():
        qs_ids, qs_attn_mask = self.pack_batch_lang(inputs)
        # qs_emb = model.get_input_embeddings()(qs_ids)
        qs_emb = model.get_model().embed_tokens(qs_ids)
        input_emb_lang = model.get_model().embed_tokens(input_ids_lang.input_ids.cuda())
        input_emb_lang = input_emb_lang.repeat(qs_emb.size(0), 1, 1)  # [b, seq_len, 4096]
        input_attn_mask = input_ids_lang.attention_mask.cuda()
        input_attn_mask = input_attn_mask.repeat(input_emb_lang.size(0), 1)  # [b, seq_len]
        input_emb_lang = torch.cat((input_emb_lang, qs_emb), dim=1)
        input_attn_mask = torch.cat((input_attn_mask, qs_attn_mask), dim=1)
        return input_emb_lang, input_attn_mask

    def get_audio_inputs(self, prompt_audio_only, inputs, model):
        input_ids_audio = self.tokenizer(
            prompt_audio_only,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )

        audios = inputs["audios"]
        audio_emb = model.get_audio_encoder()(audios["audios"].cuda(), audios["lengths"])
        input_emb_audio = model.get_model().embed_tokens(input_ids_audio.input_ids.cuda())
        input_emb_audio = input_emb_audio.repeat(audio_emb["inputs_embeds"].size(0), 1, 1)  # [b, seq_len, 4096]
        input_attn_mask = input_ids_audio.attention_mask.cuda()
        input_attn_mask = input_attn_mask.repeat(input_emb_audio.size(0), 1)  # [b, seq_len]
        input_emb_audio = torch.cat((input_emb_audio, audio_emb["inputs_embeds"]), dim=1)
        input_attn_mask = torch.cat((input_attn_mask, audio_emb["attention_mask"]), dim=1)
        return input_emb_audio, input_attn_mask

    def get_video_inputs(self, prompt_video_only, inputs, model, batch_size):
        input_ids_video = self.tokenizer(
            prompt_video_only,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )

        video_emb = model.encode_images(inputs["images"])
        video_emb = video_emb.view(batch_size, -1, video_emb.shape[-1])
        input_emb_video = model.get_model().embed_tokens(input_ids_video.input_ids.cuda())
        input_emb_video = input_emb_video.repeat(video_emb.size(0), 1, 1)  # [b, seq_len, 4096]
        input_attn_mask = input_ids_video.attention_mask.cuda()
        input_attn_mask = input_attn_mask.repeat(input_emb_video.size(0), 1)  # [b, seq_len]
        input_emb_video = torch.cat((input_emb_video, video_emb), dim=1)
        dims = (video_emb.size(0), video_emb.size(1))
        input_attn_mask = torch.cat((input_attn_mask, torch.ones(dims, dtype=torch.bool).cuda()), dim=1)
        return input_emb_video, input_attn_mask

    def pack_batch_lang(self, inputs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        split_index = (input_ids == AUDIO_TOKEN_INDEX).nonzero(as_tuple=True)

        # 提取每行 -500 以后的数据
        qs_ids = []
        qs_attn_mask = []
        for batch_idx in range(input_ids.size(0)):
            # 找到当前行的分割位置
            idxs = split_index[1][split_index[0] == batch_idx]
            if len(idxs) > 0:
                start_idx = idxs[0] + 1  # 分割后的位置
                qs_ids.append(input_ids[batch_idx, start_idx:])
                qs_attn_mask.append(attention_mask[batch_idx, start_idx:])

                # 将结果转换为张量，填充到相同长度
        qs_ids = torch.nn.utils.rnn.pad_sequence(qs_ids, batch_first=True, padding_value=0)
        qs_attn_mask = torch.nn.utils.rnn.pad_sequence(qs_attn_mask, batch_first=True, padding_value=0)

        return qs_ids, qs_attn_mask

    # Multifaceted Cycle Collaborative Debiasing
    def mccd(self, logit_qs, logit_audio, logit_video, logit_fusion):
        logit_qs = F.softmax(logit_qs.mean(dim=1), dim=-1)
        logit_audio = F.softmax(logit_audio.mean(dim=1), dim=-1)
        logit_video = F.softmax(logit_video.mean(dim=1), dim=-1)
        logit_fusion = F.softmax(logit_fusion.mean(dim=1), dim=-1)
        loss = 0

        assert (MCCD["multifaceted"]["lang"] or
                    MCCD["multifaceted"]["audio"] or
                    MCCD["multifaceted"]["video"] or
                    MCCD["cycle"]), "All of these items must not be False at the same time."

        if MCCD["multifaceted"]["lang"]:
            div_qs_fus = F.kl_div(logit_qs.log(), logit_fusion, reduction="batchmean")
            inverse_kl_div = 1 / (div_qs_fus + 1e-8)
            loss += inverse_kl_div

        if MCCD["multifaceted"]["audio"]:
            div_aud_fus = F.kl_div(logit_audio.log(), logit_fusion, reduction="batchmean")
            inverse_kl_div = 1 / (div_aud_fus + 1e-8)
            loss += inverse_kl_div

        if MCCD["multifaceted"]["video"]:
            div_vid_fus = F.kl_div(logit_video.log(), logit_fusion, reduction="batchmean")
            inverse_kl_div = 1 / (div_vid_fus + 1e-8)
            loss += inverse_kl_div
        
        if MCCD["multifaceted"]["lang"] or MCCD["multifaceted"]["audio"] or MCCD["multifaceted"]["video"]:
            print("loss_mccd_multifaced: ", loss.item())
        loss *= MCCD["lambda_multifaceted"]
        loss_mccd_cycle = 0
        if MCCD["cycle"]:
            div_qs_aud = F.kl_div(logit_qs.log(), logit_audio, reduction="batchmean")
            div_aud_vid = F.kl_div(logit_audio.log(), logit_video, reduction="batchmean")
            div_vid_qs = F.kl_div(logit_video.log(), logit_qs, reduction="batchmean")
            loss_mccd_cycle = div_qs_aud + div_aud_vid + div_vid_qs
            print("loss_mccd_cycle:",loss_mccd_cycle.item())
            loss += MCCD["lambda_cycle"] * loss_mccd_cycle

        return loss


    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [
                    name
                    for name, _ in opt_model.named_parameters()
                    if "mm_projector" in name or "vision_tower" in name
                ]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                    n in decay_parameters
                                    and n not in projector_parameters
                                    and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                    n not in decay_parameters
                                    and n not in projector_parameters
                                    and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                    n in decay_parameters
                                    and n in projector_parameters
                                    and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                    n not in decay_parameters
                                    and n in projector_parameters
                                    and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum(
                            {p.data_ptr(): p.numel() for p in module.parameters()}.values()
                        )
                        logger.info(f"skipped {module}: {skipped / 2 ** 20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped / 2 ** 20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        # print('model.model.audio_encoder.adpter.project.weight')
        # print(model.model.audio_encoder.adpter.project.weight)
        # print('model.model.audio_encoder.adpter.project.weight.requires_grad')
        # print(model.model.audio_encoder.adpter.project.weight.requires_grad)
        if self.args.lora_enable:
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            
            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(self.args.output_dir, checkpoint_folder)
            os.makedirs(output_dir, exist_ok=True)
            state_dict = get_peft_state_maybe_zero_3(
                self.model.named_parameters(), self.args.lora_bias
            )
            non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                self.model.named_parameters()
            )
            if self.args.local_rank == 0 or self.args.local_rank == -1:
                print(f"save models to {output_dir} ")
                self.model.config.save_pretrained(output_dir)
                self.model.save_pretrained(output_dir, state_dict=state_dict)
                torch.save(non_lora_state_dict, os.path.join(output_dir, 'non_lora_trainables.bin'))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)
        # if getattr(self.args, "tune_mm_mlp_adapter", False):
        #     from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

        #     checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        #     run_dir = self._get_output_dir(trial=trial)
        #     output_dir = os.path.join(run_dir, checkpoint_folder)

        #     # Only save Adapter
        #     keys_to_match = ["mm_projector", "vision_resampler"]
        #     if getattr(self.args, "use_im_start_end", False):
        #         keys_to_match.extend(["embed_tokens", "embed_in"])

        #     weight_to_save = get_mm_adapter_state_maybe_zero_3(
        #         self.model.named_parameters(), keys_to_match
        #     )

        #     if self.args.local_rank == 0 or self.args.local_rank == -1:
        #         self.model.config.save_pretrained(output_dir)
        #         torch.save(weight_to_save, os.path.join(output_dir, f"mm_projector.bin"))
        # else:
        #     super(VITATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, "tune_mm_mlp_adapter", False):
            pass
        else:
            super(VITATrainer, self)._save(output_dir, state_dict)

    def training_step(
            self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.
        Print

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        tr_loss_step = super().training_step(model, inputs)
        return tr_loss_step
        # try:
        #    #import pdb; pdb.set_trace()
        #    tr_loss_step = super().training_step(model, inputs)
        #    return tr_loss_step
        # except Exception as e:
        #    print('------------------------------------------------len of images------------------------------------------------')
        #    print(len(inputs['images']))
        #    print('------------------------------------------------input_ids------------------------------------------------')
        #    print(inputs['input_ids'].tolist())
        #    print(e)
