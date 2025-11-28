import torch

from backend import memory_management
from backend.text_processing import emphasis, parsing
from modules.shared import opts


class PromptChunk:
    def __init__(self):
        self.tokens = []
        self.multipliers = []


class GemmaTextProcessingEngine:
    def __init__(self, text_encoder, tokenizer, min_length=1):
        super().__init__()

        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

        self.min_length = min_length
        self.id_start = 2
        self.id_end = 1
        self.id_pad = 0

    def tokenize(self, texts):
        tokenized = self.tokenizer(texts, truncation=False, add_special_tokens=False)["input_ids"]
        return tokenized

    def tokenize_line(self, line):
        parsed = parsing.parse_prompt_attention(line, self.emphasis.name)

        tokenized = self.tokenize([text for text, _ in parsed])

        chunks = []
        chunk = PromptChunk()
        token_count = 0

        def next_chunk():
            nonlocal token_count
            nonlocal chunk

            chunk.tokens = [self.id_start] + chunk.tokens + [self.id_end]
            chunk.multipliers = [1.0] + chunk.multipliers + [1.0]

            current_chunk_length = len(chunk.tokens)
            token_count += current_chunk_length

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

        return chunks, token_count

    def __call__(self, texts):
        zs = []
        cache = {}

        self.emphasis = emphasis.get_current_option(opts.emphasis)()
        if any(x for x in texts if "(" in x or "[" in x) and self.emphasis.name != "Original":
            emphasis.last_extra_generation_params["Emphasis"] = self.emphasis.name

        for line in texts:
            if line != "":
                line = "You are an assistant designed to generate high-quality images with the highest degree of image-text alignment based on textual prompts. <Prompt Start> " + line

            if line in cache:
                line_z_values = cache[line]
            else:
                chunks, token_count = self.tokenize_line(line)
                line_z_values = []

                for chunk in chunks:
                    tokens = chunk.tokens
                    multipliers = chunk.multipliers

                    z = self.process_tokens(tokens, multipliers)[0]
                    line_z_values.append(z)

                # remove start/end - maybe slightly better
                count_z = len(line_z_values)
                for i in range(count_z):
                    if i - 1 >= 0 and i + 1 < count_z:
                        line_z_values[i] = line_z_values[i][1:-1]
                    elif i + 1 < count_z:
                        line_z_values[i] = line_z_values[i][:-1]
                    elif i - 1 >= 0:
                        line_z_values[i] = line_z_values[i][1:]

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

            embeds_out.append(tokens_embed)
            attention_masks.append(attention_mask)
            num_tokens.append(sum(attention_mask))

        return torch.cat(embeds_out), torch.tensor(attention_masks, device=device, dtype=torch.long), num_tokens

    def process_tokens(self, batch_tokens, batch_multipliers):
        embeds, mask, count = self.process_embeds([batch_tokens])

        if self.emphasis.name == "No norm":
            embeds *= torch.tensor(batch_multipliers).to(embeds).unsqueeze(1).unsqueeze(0)
        elif self.emphasis.name == "Original":
            original_mean = z.mean()
            embeds *= torch.tensor(batch_multipliers).to(embeds).unsqueeze(1).unsqueeze(0)
            new_mean = z.mean()
            embeds *= (original_mean / new_mean)

        z, _ = self.text_encoder(input_ids=None, embeds=embeds, attention_mask=mask, num_tokens=count)#, intermediate_output=-2, final_layer_norm_intermediate=False)

        return z
