import json
import os
import time
from itertools import repeat
import numpy as np

import torch
import torch.nn as nn
from IPython.display import FileLinks
#import coremltools as ct
from torch.nn.utils.rnn import pad_sequence
from torchmetrics import CharErrorRate
from tqdm import tqdm

from models.scripts.transformer.utils import tensor_to_word, display_attention, VECTOR_SIZE, EOS_IDX, BOS_IDX, \
    SRC_PAD_IDX, TRG_PAD_IDX
from models.scripts.utils import load_checkpoint, log_epoch, epoch_time, plot_training, Levenshtein_Normalized_distance, \
    count_postfix_violations_no_separator, count_postfix_violations_separator

# Defaults
DECODER_OUTPUT_LENGTH = 14
ENCODER_INPUT_LENGTH = 200
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1
HID_DIM = VECTOR_SIZE
ENC_PF_DIM = VECTOR_SIZE
DEC_PF_DIM = VECTOR_SIZE


# Create Model
def model_builder(
        name,
        vocab,
        device=DEVICE,
        bos_idx=BOS_IDX,
        eos_idx=EOS_IDX,
        hid_dim=HID_DIM,
        enc_heads=4,
        dec_heads=4,
        n_tokens=22,
        enc_layers=5,
        dec_layers=4,
        enc_pf_dim=ENC_PF_DIM,
        dec_pf_dim=DEC_PF_DIM * 3,
        enc_dropout=ENC_DROPOUT,
        trg_pad_idx=TRG_PAD_IDX,
        src_pad_idx=SRC_PAD_IDX,
        dec_dropout=DEC_DROPOUT,
        enc_max_length=ENCODER_INPUT_LENGTH,
        dec_max_length=DECODER_OUTPUT_LENGTH,
        **kwargs,
):
    if 'encoder' in kwargs:
        try:
            version = kwargs.get("encoder")
            hp = load_json_hypeparameters(version)
            tmp_model = model_builder(version, vocab, **hp)
            tmp_model, *_ = load_checkpoint(os.path.join("models", "check_points", f'best_model_{version}.pt'),
                                            tmp_model, strict=False)
            encoder = tmp_model.encoder
        except:
            print("Specified encoder does not exist")
            raise
    else:
        encoder = Encoder(hid_dim=hid_dim,
                          n_layers=enc_layers,
                          n_heads=enc_heads,
                          pf_dim=enc_pf_dim,
                          dropout=enc_dropout,
                          max_length=enc_max_length,
                          device=device)

    if 'decoder' in kwargs:
        try:
            version = kwargs.get("decoder")
            hp = load_json_hypeparameters(version)
            tmp_model = model_builder(version, vocab, **hp)
            tmp_model, *_ = load_checkpoint(os.path.join("models", "check_points", f'best_model_{version}.pt'),
                                            tmp_model, strict=False)
            decoder = tmp_model.decoder
        except:
            print("Specified decoder does not exist")
            raise
    else:
        decoder = Decoder(output_dim=n_tokens,
                          hid_dim=hid_dim,
                          n_layers=dec_layers,
                          n_heads=dec_heads,
                          pf_dim=dec_pf_dim,
                          dropout=dec_dropout,
                          max_length=dec_max_length,
                          device=device)
    model = Transformer(encoder=encoder,
                        decoder=decoder,
                        src_pad_idx=src_pad_idx,
                        trg_pad_idx=trg_pad_idx,
                        bos_idx=bos_idx,
                        eos_idx=eos_idx,
                        device=device,
                        vocab=vocab,
                        name=name).to(device)
    return model


def load_json_hypeparameters(version: str):
    with open(os.path.join("models", "hyperparameters", version + ".json"), "r+") as hpf:
        hp = json.load(hpf)
    return hp


# Create Model

def pad_collate_fn(batch):
    """Pad and collate batches"""

    xx, yy = [], []
    padding_tensor = torch.tensor([TRG_PAD_IDX])

    for (x, y) in batch:
        xx.append(x)
        diff = (DECODER_OUTPUT_LENGTH - (y.shape[0])) - 2
        y_ = torch.cat([torch.tensor([BOS_IDX]),
                        y, torch.tensor([EOS_IDX]), torch.tensor(list(repeat(padding_tensor, diff)))], dim=0)

        # Because equally-padded tensors have float
        yy.append(torch.tensor(y_, dtype=torch.int64))

    # Pad sequence to even size
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=TRG_PAD_IDX)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=TRG_PAD_IDX)

    return xx_pad, yy_pad


def make_src_mask(src, src_pad_idx=SRC_PAD_IDX, device=DEVICE):
    pad_tensor = torch.zeros(src.shape[1:], device=device) + src_pad_idx
    src_mask = torch.logical_not(torch.all(torch.eq(pad_tensor, src), dim=2)).unsqueeze(1).unsqueeze(2)
    return src_mask


def make_trg_mask(trg, trg_pad_idx=TRG_PAD_IDX, device=DEVICE):
    trg_pad_mask = (trg != trg_pad_idx).unsqueeze(1).unsqueeze(2)
    trg_len = trg.shape[1]
    trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=device)).bool()
    trg_mask = trg_pad_mask & trg_sub_mask
    return trg_mask


# Transformer Implementation

class Transformer(nn.Module):
    def __init__(
            self,
            encoder,
            decoder,
            src_pad_idx,
            trg_pad_idx,
            bos_idx,
            eos_idx,
            device,
            vocab,
            name
    ):
        super().__init__()

        self.device = device
        self.encoder = encoder
        self.decoder = decoder
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.name = name
        self.vocab = vocab
        self.log_path = os.path.join("models", "logs", f'{self.name}.log')
        self.bm_path = os.path.join("models", "check_points", f'best_model_{self.name}.pt')

    def forward(self, src, trg, src_mask, trg_mask):
        enc_src, enc_attn = self.encoder(src, src_mask)
        return self.decoder(trg, enc_src, trg_mask, src_mask), enc_attn

    def count_parameters(self):
        count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'The model has {count:,} trainable parameters.')

    def save_hyperparameters_to_json(self):
        hp = {'enc_heads': self.encoder.n_heads, 'dec_heads': self.decoder.n_heads, 'enc_layers': self.encoder.n_layers,
              'dec_layers': self.decoder.n_layers, 'enc_dropout': 0.1, 'dec_dropout': 0.1, 'bos_idx': self.bos_idx,
              'eos_idx': self.eos_idx, 'n_tokens': self.decoder.output_dim, 'hid_dim': self.encoder.hid_dim,
              'trg_pad_idx': self.trg_pad_idx, 'vector_size': VECTOR_SIZE, 'src_pad_idx': self.src_pad_idx,
              'enc_pf_dim': self.encoder.pf_dim, 'dec_pf_dim': self.decoder.pf_dim, 'n_features': 2,
              'dec_max_length': self.decoder.max_length, 'enc_max_length': self.encoder.max_length,
              'vocab': {self.vocab.itos[i]: i for i in range(self.decoder.output_dim)}}
        with open(os.path.join("models", "hyperparameters", self.name + ".json"), "w+") as hpf:
            json.dump(hp, hpf)

    def load_best_version(self):
        self.load_state_dict(torch.load(self.bm_path, map_location=torch.device(self.device))['state_dict'])

    def load_checkpoint(self, strict=True, optimizer=None, scheduler=None):
        """Load model and optimizer from a checkpoint"""

        checkpoint = torch.load(self.bm_path, map_location=torch.device(self.device))
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])

        if scheduler:
            scheduler.load_state_dict(checkpoint['scheduler'])

        self.load_state_dict(checkpoint['state_dict'], strict=strict)
        return optimizer, scheduler

    def plot_training(self):
        plot_training(self.log_path)

    def trace_and_export(self, src, trg, version):

        # Export Encoder to CoreML (direct route)
        output_dir = "models/exports"
        src = src.to('cpu')
        trg = trg.to('cpu')
        src_mask = make_src_mask(src).int().to('cpu')
        print(src.dtype, src_mask.dtype)
        print(src.shape, src_mask.shape)
        model_input = (src, src_mask)

        with torch.no_grad():
            self.encoder.eval()
            traced_model = torch.jit.trace(self.encoder.to('cpu'), model_input)

        ml_model = ct.convert(
            traced_model,
            inputs=[
                ct.TensorType(name="src", shape=model_input[0].shape, dtype=float),
                ct.TensorType(name="src_mask", shape=model_input[1].shape, dtype=bool)
            ],
            #     minimum_deployment_target=ct.target.iOS15,
        )

        # rename output description and save
        spec = ml_model.get_spec()
        output_desc = list(map(lambda x: '%s' % x.name, ml_model.output_description._fd_spec))
        ct.utils.rename_feature(spec, output_desc[0], 'src_enc')
        ct.utils.rename_feature(spec, output_desc[1], 'self_attentions')
        ml_model = ct.models.MLModel(spec)
        ml_model.save(os.path.join(output_dir, f'StrokeSequenceEncoder_', version))

        # Export Decoder to CoreML (direct route)
        trg_mask = make_trg_mask(trg)
        enc_src, _ = self.encoder(src, src_mask).to('cpu')
        model_input = (trg.int(), enc_src, trg_mask, src_mask)

        with torch.no_grad():
            self.decoder.eval()
            traced_model = torch.jit.trace(self.decoder, model_input)

        ml_model = ct.convert(
            traced_model,
            inputs=[
                ct.TensorType(name="trg", shape=model_input[0].shape, dtype=int),
                ct.TensorType(name="enc_src", shape=model_input[1].shape, dtype=float),
                ct.TensorType(name="trg_mask", shape=model_input[2].shape, dtype=bool),
                ct.TensorType(name="src_mask", shape=model_input[3].shape, dtype=bool)
            ]
        )
        # rename output description and save
        spec = ml_model.get_spec()
        output_desc = list(map(lambda x: '%s' % x.name, ml_model.output_description._fd_spec))
        ct.utils.rename_feature(spec, output_desc[0], 'out_dec')
        ct.utils.rename_feature(spec, output_desc[1], 'cross_attentions')
        ct.utils.rename_feature(spec, output_desc[2], 'self_attentions')
        ml_model = ct.models.MLModel(spec)
        ml_model.save(os.path.join(output_dir, f'StrokeSequenceDecoder_', version))

        # List files
        FileLinks(os.path.join(output_dir))

    def train_f(self, iterator, optimizer, criterion, clip, scheduler=None):

        self.train()

        epoch_loss = 0

        for i, batch in enumerate(iterator):
            src = batch[0].to(self.device)
            trg = batch[1].to(self.device)

            src_mask = make_src_mask(src, SRC_PAD_IDX, self.device)
            trg_mask = make_trg_mask(trg[:, :-1], TRG_PAD_IDX, self.device)

            optimizer.zero_grad()

            dec, enc_attn = self.forward(src, trg[:, :-1], src_mask, trg_mask)  # Remove last (eos or pad)
            output = dec[0]
            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)  # Remove first (bos)

            loss = criterion(output, trg)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.parameters(), clip)

            optimizer.step()

            epoch_loss += loss.item()

        if scheduler:
            scheduler.step()

        return epoch_loss / len(iterator)

    def evaluate_f(self, iterator, criterion=nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)):

        self.eval()

        epoch_loss = 0

        with torch.no_grad():
            for i, batch in enumerate(iterator):
                src = batch[0].to(self.device)
                trg = batch[1].to(self.device)

                src_mask = make_src_mask(src, self.src_pad_idx, self.device)
                trg_mask = make_trg_mask(trg[:, :-1], self.trg_pad_idx, self.device)

                dec, enc_attn = self.forward(src, trg[:, :-1], src_mask, trg_mask)  # Remove last (eos or pad)
                output = dec[0]
                output_dim = output.shape[-1]

                output = output.contiguous().view(-1, output_dim)
                trg = trg[:, 1:].contiguous().view(-1)  # Remove first (bos)

                loss = criterion(output, trg)

                epoch_loss += loss.item()

        return epoch_loss / len(iterator)

    def train_loop(self,
                   train_set,
                   valid_set,
                   resume=False,
                   optimizer=None,
                   scheduler=None,
                   n_epochs=8000,
                   criterion=nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX),
                   clip=1
                   ):
        best_valid_loss = float('inf')

        if resume:
            if os.path.exists(self.bm_path):
                print(f"Loaded previous model '{self.bm_path}'\n")
                optimizer, scheduler = self.load_checkpoint(optimizer=optimizer, scheduler=scheduler)
            else:
                print("Cannot resume training: no weights file found")

        else:
            if not optimizer:
                optimizer = torch.optim.Adam(self.parameters(), lr=0.003)

        print(f"Training started using device: {self.device}")

        for epoch in range(n_epochs):

            start_time = time.time()

            train_loss = self.train_f(train_set, optimizer, criterion, clip, scheduler)
            valid_loss = self.evaluate_f(valid_set, criterion)

            log_epoch(self.log_path, epoch, train_loss, valid_loss)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            # Save only best model
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss

                checkpoint = {
                    'vocab': self.vocab,
                    'state_dict': self.state_dict(),
                    'optimizer': optimizer.state_dict()
                }

                torch.save(checkpoint, self.bm_path)

            print(f'Epoch: {epoch + 1:02}/{n_epochs} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f}')

    def predict(self, src, max_length=None):
        self.eval()
        if max_length is None:
            max_length = self.decoder.max_length
        src_mask = make_src_mask(src, self.src_pad_idx, self.device)

        with torch.no_grad():
            enc_src, enc_self_attention = self.encoder(src, src_mask)

        # Start with a beginning of sequence token
        trg_indexes = [self.bos_idx]

        for i in range(max_length):
            trg = torch.tensor(trg_indexes, dtype=torch.int64).unsqueeze(0).to(self.device)
            trg_mask = make_trg_mask(trg, self.trg_pad_idx, self.device)

            with torch.no_grad():
                output, cross_attention, dec_self_attention = self.decoder(trg, enc_src, trg_mask, src_mask)

            pred_token = output.argmax(2)[:, -1].item()
            trg_indexes.append(pred_token)

            if pred_token == self.eos_idx:
                break

        trg_indexes = trg_indexes[1:]  # Exclude '<bos>'

        trg_tokens = [self.vocab.itos[i] for i in trg_indexes]

        return trg_tokens, (cross_attention, enc_self_attention, dec_self_attention)

    def evaluate_Levenshtein_accuracy(self, t_set):
        loss = 0
        count = 0
        for b_x, b_y in tqdm(t_set):
            b_x = b_x.to(self.device)
            b_y = b_y.to(self.device)
            assert len(b_x) == len(b_y), "Mismatch in test dimensions"
            for x_i, y_i in zip(b_x, b_y):
                prediction, attention = self.predict(src=x_i.unsqueeze(0))
                gt = tensor_to_word(y_i, self.vocab)
                dis = Levenshtein_Normalized_distance(a=''.join(gt).strip('<bos>').strip('<pad>').strip('<eos>'),
                                                      b=''.join(prediction).strip('<bos>').strip('<eos>').strip(
                                                          '<pad>'))
                count += 1
                loss = loss + ((dis - loss) / count)
        return 1 - loss

    def evaluate_CER(self, t_set):
        loss = 0
        count = 0
        metric = CharErrorRate()
        for b_x, b_y in tqdm(t_set):
            b_x = b_x.to(self.device)
            b_y = b_y.to(self.device)
            assert len(b_x) == len(b_y), "Mismatch in test dimensions"
            for x_i, y_i in zip(b_x, b_y):
                prediction, attention = self.predict(x_i.unsqueeze(0))
                gt = tensor_to_word(y_i, self.vocab)

                dis = metric(''.join(gt).strip('<bos>').strip('<pad>').strip('<eos>'),
                             ''.join(prediction).strip('<bos>').strip('<eos>').strip('<pad>'))
                count += 1
                loss = loss + ((dis - loss) / count)
        return loss

    def evaluate_postfix_accuracy(self, t_set, separator=False):
        min_loss = 0
        max_loss = 0
        count = 0
        for b_x, b_y in tqdm(t_set):
            b_x = b_x.to(self.device)
            b_y = b_y.to(self.device)
            assert len(b_x) == len(b_y), "Mismatch in test dimensions"
            for x_i in b_x:
                prediction, attention = self.predict(x_i.unsqueeze(0))
                if separator:
                    errors = count_postfix_violations_separator(''.join(prediction).strip('<bos>')
                                                                   .strip('<eos>').strip('<pad>'))
                else:
                    errors = count_postfix_violations_no_separator(''.join(prediction).strip('<bos>')
                                                               .strip('<eos>').strip('<pad>'))
                count += 1
                max_loss = max_loss + ((errors - max_loss) / count)
                inc = 1 if errors > 0 else 0
                min_loss = min_loss + ((inc - min_loss) / count)
        return 1 - max_loss, 1 - min_loss

    def display_attention(self, raw_input, pred, attention):
        display_attention(raw_input, pred, attention, n_heads=self.decoder.n_heads, n_rows=self.decoder.n_heads, exclude_paddings=True)

    def predict_with_probabilities(self, src, max_length=None):

        self.eval()
        if max_length is None:
            max_length = self.decoder.max_length
        src_mask = make_src_mask(src, self.src_pad_idx, self.device)
        probabilities = torch.zeros((max_length, self.decoder.output_dim))

        with torch.no_grad():
            enc_src, enc_self_attention = self.encoder(src, src_mask)

        # Start with a beginning of sequence token
        trg_indexes = [self.bos_idx]

        for i in range(max_length):
            trg = torch.tensor(trg_indexes, dtype=torch.int64).unsqueeze(0).to(self.device)
            trg_mask = make_trg_mask(trg, self.trg_pad_idx, self.device)

            with torch.no_grad():
                output, cross_attention, dec_self_attention = self.decoder(trg, enc_src, trg_mask, src_mask)

                soft = nn.Softmax(dim=2)
                output = soft(output)

            probabilities[i, :] = output[0, -1, :]
            pred_token = output.argmax(2)[:, -1].item()

            trg_indexes.append(pred_token)

            if pred_token == EOS_IDX:
                break
        trg_indexes = trg_indexes[1:]  # Exclude '<bos>'

        trg_tokens = [self.vocab.itos[i] for i in trg_indexes]

        return trg_tokens, (cross_attention, enc_self_attention, dec_self_attention), probabilities

    def compute_glyph_average_probabilities(self, t_set):
        avg_probabilities = np.zeros((self.decoder.output_dim, self.decoder.output_dim))
        counts = np.zeros(self.decoder.output_dim)
        for b_x, b_y in tqdm(t_set):
            b_x = b_x.to(self.device)
            b_y = b_y.to(self.device)
            assert len(b_x) == len(b_y), "Mismatch in test dimensions"
            for x_i, y_i in zip(b_x, b_y):
                prediction, attention, probabilities = self.predict_with_probabilities(x_i.unsqueeze(0))
                gt_list = [i for i in y_i.tolist() if i != 1]
                for i, elem in enumerate(gt_list[1:], start=0):
                    counts[elem] += 1
                    avg_probabilities[elem, :] = np.add(avg_probabilities[elem, :], (
                        np.subtract(np.array(probabilities[i, :]), avg_probabilities[elem, :])) / counts[elem])
        return avg_probabilities


# Decoder Implementation
class Decoder(nn.Module):
    def __init__(
            self,
            output_dim,
            hid_dim,
            n_layers,
            n_heads,
            pf_dim,
            dropout,
            max_length,
            device
    ):
        super().__init__()
        self.device = device
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.max_length = max_length
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.layers = nn.ModuleList(
            [DecoderLayer(
                hid_dim,
                n_heads,
                pf_dim,
                dropout,
                device) for _ in range(n_layers)]
        )
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        trg_len = trg.shape[1]
        batch_size = trg.shape[0]
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
        cross_attentions = []
        self_attentions = []
        for layer in self.layers:
            trg, cross_attention, self_attention = layer(trg, enc_src, trg_mask, src_mask)
            cross_attentions.append(cross_attention)
            self_attentions.append(self_attention)
        output = self.fc_out(trg)
        return output, torch.cat(cross_attentions), torch.cat(self_attentions)


# DecoderLayer Implementation

class DecoderLayer(nn.Module):
    def __init__(
            self,
            hid_dim,
            n_heads,
            pf_dim,
            dropout,
            device
    ):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        _trg, self_attention = self.self_attention(trg, trg, trg, trg_mask)
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
        _trg, cross_attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
        _trg = self.positionwise_feedforward(trg)
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        return trg, cross_attention, self_attention


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(torch.relu(self.fc_1(x)))
        x = self.fc_2(x)
        return x


# Encoder Implementation

class Encoder(nn.Module):
    """Encoder is a stack of n_layers EncoderLayer"""

    def __init__(
            self,
            hid_dim,
            n_layers,
            n_heads,
            pf_dim,
            dropout,
            max_length,
            device
    ):
        super().__init__()
        self.device = device
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.max_length = max_length
        self.layers = nn.ModuleList(
            [EncoderLayer(
                hid_dim,
                n_heads,
                pf_dim,
                dropout,
                device) for _ in range(n_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src, src_mask):
        src_len = src.shape[1]
        batch_size = src.shape[0]
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        src = self.dropout((src * self.scale) + self.pos_embedding(pos))
        self_attentions = []
        for layer in self.layers:
            src, self_attention = layer(src, src_mask)
            self_attentions.append(self_attention)
        return src, torch.cat(self_attentions)


# EncoderLayer Implementation

class EncoderLayer(nn.Module):
    def __init__(
            self,
            hid_dim,
            n_heads,
            pf_dim,
            dropout,
            device
    ):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        _src, self_attention = self.self_attention(src, src, src, src_mask)
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        _src = self.positionwise_feedforward(src)
        src = self.ff_layer_norm(src + self.dropout(_src))
        return src, self_attention


# MultiHeadAttentionLayer Implementation

class MultiHeadAttentionLayer(nn.Module):
    def __init__(
            self,
            hid_dim,
            n_heads,
            dropout,
            device
    ):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # Q = [batch size, query len, hid dim]
        # K = [batch size, key len, hid dim]
        # V = [batch size, value len, hid dim]
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # Q = [batch size, query len, hid dim]
        # K = [batch size, key len, hid dim]
        # V = [batch size, value len, hid dim]
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 3, 1)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Q = [batch size, n heads, query len, head dim]
        # K = [batch size, n heads, key len, head dim]
        # V = [batch size, n heads, value len, head dim]
        energy = torch.matmul(Q, K) / self.scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)
        x = torch.matmul(self.dropout(attention), V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hid_dim)
        x = self.fc_o(x)

        return x, attention
