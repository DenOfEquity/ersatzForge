import torch

from backend.text_processing import parsing, emphasis
from backend import memory_management

from modules.shared import opts


class PromptChunk:
    def __init__(self):
        self.tokens = []
        self.multipliers = []


class T5TextProcessingEngine:
    def __init__(self, text_encoder, tokenizer, min_length=256, end_with_pad=False, add_special_tokens=False):
        super().__init__()

        self.text_encoder = text_encoder.transformer
        self.tokenizer = tokenizer

        self.min_length = min_length
        self.end_with_pad = end_with_pad
        self.add_special_tokens = add_special_tokens
        self.id_end = 1
        self.id_pad = 0

        vocab = self.tokenizer.get_vocab()

        self.comma_token = vocab.get(',</w>', None)

    def tokenize(self, texts):
        tokenized = self.tokenizer(texts, truncation=False, add_special_tokens=self.add_special_tokens)["input_ids"]
        return tokenized

    def _process_tokens(self, tokens):
        attention_masks = []

        for x in tokens:
            attention_mask = []
            eos = False

            for y in x:
                if isinstance(y, int):
                    attention_mask.append(0 if eos else 1)
                    if not eos and int(y) == self.id_end:
                        eos = True

            attention_masks.append(attention_mask)

        return torch.tensor(attention_masks, dtype=torch.long)

    def encode_with_transformers(self, tokens, attention_mask=None):
        device = memory_management.text_encoder_device()
        tokens = tokens.to(device)
        self.text_encoder.shared.to(device=device, dtype=torch.float32)

        if attention_mask is not None:
            z = self.text_encoder(input_ids=tokens, attention_mask=attention_mask)
        else:
            z = self.text_encoder(input_ids=tokens,)

        return z

    def tokenize_line(self, line):
        parsed = parsing.parse_prompt_attention(line, self.emphasis.name)

        tokenized = self.tokenize([text for text, _ in parsed])

        chunks = []
        chunk = PromptChunk()

        def next_chunk():
            nonlocal chunk

            if self.end_with_pad:
                chunk.tokens = chunk.tokens + [self.id_pad]
                chunk.multipliers = chunk.multipliers + [1.0]

            chunk.tokens = chunk.tokens + [self.id_end]
            chunk.multipliers = chunk.multipliers + [1.0]
            current_chunk_length = len(chunk.tokens)

            remaining_count = self.min_length - current_chunk_length

            if remaining_count > 0:
                chunk.tokens += [self.id_pad] * remaining_count
                chunk.multipliers += [1.0] * remaining_count

            chunks.append(chunk)
            chunk = PromptChunk()

        for tokens, (text, weight) in zip(tokenized, parsed):
            if text == 'BREAK' and weight == -1:
                next_chunk()
                continue

            position = 0
            while position < len(tokens):
                token = tokens[position]
                chunk.tokens.append(token)
                chunk.multipliers.append(weight)
                position += 1

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

                    z = self.process_tokens([tokens], [multipliers])[0]
                    line_z_values.append(z)

                count_z = len(line_z_values) - 1 # number of chunks to shorten, starting from first
                if count_z:
                    for i in range(count_z):
                        if self.end_with_pad:
                            line_z_values[i] = line_z_values[i][:-2]
                        else:
                            line_z_values[i] = line_z_values[i][:-1]

                    line_z_values = [torch.cat(line_z_values, dim=0, )]

                cache[line] = line_z_values

            zs.extend(line_z_values)

        return zs
        # pad zs
        # max_length = len(max(zs, key=len))
        # for i in range(len(zs)):
            # pad = max_length - len(zs[i])
            # if pad > 0:
                # zs[i] = torch.cat([zs[i], zs[i].new_zeros([pad, zs[i].shape[1]])])

        # return torch.stack(zs)

    def process_tokens(self, batch_tokens, batch_multipliers):
        if self.text_encoder.config["model_type"] == "umt5":
            attention_mask = self._process_tokens(batch_tokens)
            tokens = torch.asarray(batch_tokens)
            z = self.encode_with_transformers(tokens, attention_mask)
        else:
            tokens = torch.asarray(batch_tokens)
            z = self.encode_with_transformers(tokens)

        self.emphasis.tokens = batch_tokens
        self.emphasis.multipliers = torch.asarray(batch_multipliers).to(z)
        self.emphasis.z = z
        self.emphasis.after_transformers()
        z = self.emphasis.z

        if self.text_encoder.config["model_type"] == "umt5":
            z *= attention_mask.unsqueeze(-1).to(z)

        return z
