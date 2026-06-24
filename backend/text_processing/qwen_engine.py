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
    def __init__(self, text_encoder, tokenizer, is_flux2=False, is_ernie=False, is_krea2=False):
        super().__init__()

        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.is_flux2 = is_flux2
        self.is_ERNIE = is_ernie
        self.is_krea2 = is_krea2

        self.id_pad = 0 if is_ernie else 151643
        # self.min_length = 512 if is_flux2 else 1 #flux min 512? or pow2
        # self.llama_template = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        
        if is_flux2:
            self.intermediate_output = [9, 18, 27]
        elif is_krea2:
            self.intermediate_output = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]
        else:
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

            if self.is_flux2:
                #             <|im_start|>user\n                  <|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
                chunk.tokens = [151644, 872, 198] + chunk.tokens + [151645, 198, 151644, 77091, 198, 151667, 198, 198, 151668, 198, 198]
                chunk.multipliers = [1.0, 1.0, 1.0] + chunk.multipliers + [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            elif self.is_krea2:
                #"<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|\n    <|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
                #             <|im_start|>user\n                  <|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
                chunk.tokens = [151644, 872, 198] + chunk.tokens + [151645, 198, 151644, 77091, 198, 151667, 198, 198, 151668, 198, 198]
                chunk.multipliers = [1.0, 1.0, 1.0] + chunk.multipliers + [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            elif self.is_ERNIE:
                chunk.tokens = [1] + chunk.tokens
                chunk.multipliers = [1.0] + chunk.multipliers
            else:
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
                if self.is_flux2:
                    s = 3
                    e = -11
                elif self.is_krea2: # update based on template
                    s = 3
                    e = -11
                elif self.is_ERNIE:
                    s = 1
                    e = None
                else:
                    s = 3
                    e = -5
                count_z = len(line_z_values)
                for i in range(count_z):
                    if i - 1 >= 0 and i + 1 < count_z:
                        line_z_values[i] = line_z_values[i][s:e]
                    elif i + 1 < count_z:
                        line_z_values[i] = line_z_values[i][:e]
                    elif i - 1 >= 0:
                        line_z_values[i] = line_z_values[i][s:]

                line_z_values = [torch.cat(line_z_values, dim=0, )]

                cache[line] = line_z_values

            zs.extend(line_z_values)

        return zs

    def process_embeds(self, batch_tokens):
        torch_device = memory_management.get_torch_device()
        device = memory_management.text_encoder_device()

        embeds_out = []
        attention_masks = []
        num_tokens = []

        for tokens in batch_tokens:
            attention_mask = []
            tokens_temp = []

            for t in tokens:
                token = int(t)
                attention_mask.append(0 if token == self.id_pad else 1)
                tokens_temp += [token]

            tokens_embed = torch.tensor([tokens_temp], device=device, dtype=torch.long)
            tokens_embed = self.text_encoder.get_input_embeddings()(tokens_embed)

            embeds_out.append(tokens_embed)
            attention_masks.append(attention_mask)
            num_tokens.append(sum(attention_mask))

        return torch.cat(embeds_out).to(device=torch_device), torch.tensor(attention_masks, device=torch_device, dtype=torch.long), num_tokens

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

        memory_management.soft_empty_cache()

        if self.is_flux2 or self.is_krea2:
            z = z.movedim(1, 2)
            z = z.reshape(z.shape[0], z.shape[1], -1)
        return z
