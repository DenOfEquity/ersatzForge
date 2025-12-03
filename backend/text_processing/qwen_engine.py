# https://github.com/comfyanonymous/ComfyUI/blob/v0.3.75/comfy/sd1_clip.py
# https://github.com/comfyanonymous/ComfyUI/blob/v0.3.75/comfy/text_encoders/z_image.py

#via ForgeNeo by Haoming02
# added emphasis, combining of chunks after BREAK


import torch

from backend import memory_management
from backend.text_processing import emphasis, parsing
from modules.shared import opts


class PromptChunk:
    def __init__(self):
        self.tokens = []
        self.multipliers = []


class Qwen3TextProcessingEngine:
    def __init__(self, text_encoder, tokenizer):
        super().__init__()

        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

        self.id_pad = 151643
        # self.llama_template = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        self.intermediate_output = -2
        self.layer_norm_hidden_state = False

    def tokenize(self, texts):
        return self.tokenizer(texts)["input_ids"]

    def tokenize_line(self, line):
        parsed = parsing.parse_prompt_attention(line, self.emphasis.name)

        tokenized = self.tokenize([text for text, _ in parsed])

        chunks = []
        chunk = PromptChunk()

        def next_chunk():
            nonlocal chunk

            #             <|im_start|>user\n                  <|im_end|>\n<|im_start|>assistant\n
            chunk.tokens = [151644, 872, 198] + chunk.tokens + [151645, 198, 151644, 77091, 198]
            chunk.multipliers = [1.0, 1.0, 1.0] + chunk.multipliers + [1.0, 1.0, 1.0, 1.0, 1.0]

            chunks.append(chunk)
            chunk = PromptChunk()

        for tokens, (text, weight) in zip(tokenized, parsed):
            if text == "BREAK" and weight == -1:
                next_chunk()
                continue

            chunk.tokens.extend(tokens)
            chunk.multipliers.extend([weight] * len(tokens))

        if chunk.tokens or not chunks:
            next_chunk()

        return chunks

    def __call__(self, texts):
        zs = []
        cache = {}

        self.emphasis = emphasis.get_current_option(opts.emphasis)()
        if any(x for x in texts if "(" in x or "[" in x) and self.emphasis.name != "Original":
            emphasis.last_extra_generation_params["Emphasis"] = self.emphasis.name

        for line in texts:
            if line in cache:
                line_z_values = cache[line]
            else:
                chunks = self.tokenize_line(line)
                line_z_values = []

                for chunk in chunks:
                    tokens = chunk.tokens
                    multipliers = chunk.multipliers

                    z = self.process_tokens(tokens, multipliers)[0]
                    line_z_values.append(z)

                # remove start/end tokens
                count_z = len(line_z_values)
                for i in range(count_z):
                    if i - 1 >= 0 and i + 1 < count_z:
                        line_z_values[i] = line_z_values[i][3:-5]
                    elif i + 1 < count_z:
                        line_z_values[i] = line_z_values[i][:-5]
                    elif i - 1 >= 0:
                        line_z_values[i] = line_z_values[i][3:]

                line_z_values = [torch.cat(line_z_values, dim=0, )]
                cache[line] = line_z_values

            zs.extend(line_z_values)

        return zs

    def process_embeds(self, batch_tokens):
        device = memory_management.text_encoder_device()

        embeds_out = []
        attention_masks = []
        num_tokens = []

        for tokens in batch_tokens:
            attention_mask = []
            tokens_temp = []
            eos = False
            index = 0

            for t in tokens:
                token = int(t)
                attention_mask.append(0 if eos else 1)
                tokens_temp += [token]
                if not eos and token == self.id_pad:
                    eos = True
                index += 1

            tokens_embed = torch.tensor([tokens_temp], device=device, dtype=torch.long)
            tokens_embed = self.text_encoder.get_input_embeddings()(tokens_embed)

            index = 0

            embeds_out.append(tokens_embed)
            attention_masks.append(attention_mask)
            num_tokens.append(sum(attention_mask))

        return torch.cat(embeds_out), torch.tensor(attention_masks, device=device, dtype=torch.long), num_tokens

    def process_tokens(self, batch_tokens, batch_multipliers):
        embeds, mask, count = self.process_embeds([batch_tokens])

        if self.emphasis.name == "No norm":
            embeds *= torch.tensor(batch_multipliers).to(embeds)[None, :, None]
        elif self.emphasis.name == "Original":
            original_mean = embeds.mean()
            embeds *= torch.tensor(batch_multipliers).to(embeds)[None, :, None]
            new_mean = embeds.mean()
            embeds *= (original_mean / new_mean)

        _, z = self.text_encoder(
            None,
            attention_mask=mask,
            embeds=embeds,
            num_tokens=count,
            intermediate_output=self.intermediate_output,
            final_layer_norm_intermediate=self.layer_norm_hidden_state,
        )

        return z
