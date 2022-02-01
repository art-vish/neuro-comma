from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Type, Union, overload
from unittest import result

import torch
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from neuro_comma.dataset import BaseDataset
from neuro_comma.model import CorrectionModel, DeepPunctuation
from neuro_comma.pretrained import PRETRAINED_MODELS, TOKEN_IDX
from neuro_comma.utils import get_last_pretrained_weight_path, load_params


class BasePredictor:
    def __init__(self,                 
                 targets: dict[str, int],
                 punctuation_dict: dict[str, int],
                 pretrained_model: str,
                 model_weights_path: str,
                 pretrained_model_path: Optional[str] = None,
                 pretrained_tokenizer_model_path: Optional[str] = None,
                 dataset_class: Type[BaseDataset] = BaseDataset,
                 quantization: Optional[bool] = False,
                 freeze_pretrained: Optional[bool] = False,
                 lstm_dim: Optional[int] = -1,
                 sequence_length: Optional[int] = 256,
                 *args,
                 **kwargs,
                 ) -> None:

        self.device = torch.device('cuda' if (
            not quantization) and torch.cuda.is_available() else 'cpu')
        self.targets = targets
        self.punctuation_dict = punctuation_dict
        self.pretrained_model = pretrained_model
        self.pretrained_model_path = str(
            pretrained_model_path) if pretrained_model_path else pretrained_model
        self.pretrained_tokenizer_model_path = str(
            pretrained_tokenizer_model_path) if pretrained_tokenizer_model_path else pretrained_model
        self.weights = model_weights_path
        self.freeze_pretrained = freeze_pretrained
        self.lstm_dim = lstm_dim
        self.sequence_length = sequence_length

        self.model = self.load_model(quantization=quantization)
        self.tokenizer = self.load_tokenizer()
        self.dataset_class = dataset_class

    @overload
    def load_model(self, quantization: Optional[bool] = False) -> CorrectionModel:       
        ...
        

    def load_tokenizer(self) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
        tokenizer = PRETRAINED_MODELS[self.pretrained_model][1].from_pretrained(
            self.pretrained_tokenizer_model_path)
        return tokenizer


class RepunctPredictor(BasePredictor):
    
    def load_model(self, quantization: Optional[bool] = False) -> CorrectionModel:
        model = CorrectionModel(self.pretrained_model,
                                self.pretrained_model_path,
                                self.punctuation_dict,
                                self.freeze_pretrained,
                                self.lstm_dim)

        if quantization:
            model = model.quantize()

        model.to(self.device)
        model.load(self.weights, map_location=self.device)        
        model.eval()
        return model
    
    def __call__(self, text: str) -> str:       
        
        words_original_case = text.split()
        tokens = text.split()
        result = ""

        token_style = PRETRAINED_MODELS[self.pretrained_model][3]
        seq_len = self.sequence_length
        decode_idx = 0

        data = torch.tensor(self.dataset_class.parse_tokens(tokens,
                                                            self.tokenizer,
                                                            seq_len,
                                                            token_style))

        x_indecies = torch.tensor([0])
        x = torch.index_select(data, 1, x_indecies).reshape(
            2, -1).to(self.device)

        attn_mask_indecies = torch.tensor([2])
        attn_mask = torch.index_select(
            data, 1, attn_mask_indecies).reshape(2, -1).to(self.device)

        y_indecies = torch.tensor([4])
        y_mask = torch.index_select(data, 1, y_indecies).view(-1)

        with torch.no_grad():
            y_predict = self.model(x, attn_mask)
            y_predict = y_predict.view(-1, y_predict.shape[2])
            y_predict = torch.argmax(y_predict, dim=1).view(-1)

        for i in range(y_mask.shape[0]):
            if y_mask[i] == 1:
                result += words_original_case[decode_idx]
                result += self.targets[y_predict[i].item()]
                result += ' '
                decode_idx += 1

        result = result.strip()
        return result

# Для предсказания на основе моделей https://github.com/xashru/punctuation-restoration
class RepunctPredictorOriginal(BasePredictor):
    def load_model(self, quantization: Optional[bool] = False) -> CorrectionModel:
        model = DeepPunctuation(self.pretrained_model,
                                self.pretrained_model_path,
                                self.punctuation_dict,
                                self.freeze_pretrained,
                                self.lstm_dim)
        model.to(self.device)
        model.load_state_dict(torch.load(self.weights, map_location=self.device), strict=False)
        model.eval()
        
        return model
    
    def __call__(self, text: str) -> str:        
        words = text.lower().split() if text else []
        word_pos = 0
        result = ""
        decode_idx = 0
        token_style = PRETRAINED_MODELS[self.pretrained_model][3]
        while word_pos < len(words):
                x = [TOKEN_IDX[token_style]['START_SEQ']]
                y_mask = [0]

                while len(x) < self.sequence_length and word_pos < len(words):
                    tokens = self.tokenizer.tokenize(words[word_pos])
                    if len(tokens) + len(x) >= self.sequence_length:
                        break
                    else:
                        for i in range(len(tokens) - 1):
                            x.append(
                                self.tokenizer.convert_tokens_to_ids(tokens[i]))
                            y_mask.append(0)
                        x.append(
                            self.tokenizer.convert_tokens_to_ids(tokens[-1]))
                        y_mask.append(1)
                        word_pos += 1
                x.append(TOKEN_IDX[token_style]['END_SEQ'])
                y_mask.append(0)
                if len(x) < self.sequence_length:
                    x = x + [TOKEN_IDX[token_style]['PAD']
                        for _ in range(self.sequence_length - len(x))]
                    y_mask = y_mask + \
                        [0 for _ in range(self.sequence_length - len(y_mask))]
                attn_mask = [1 if token != TOKEN_IDX[token_style]
                    ['PAD'] else 0 for token in x]

                x = torch.tensor(x).reshape(1, -1)
                y_mask = torch.tensor(y_mask)
                attn_mask = torch.tensor(attn_mask).reshape(1, -1)
                x, attn_mask, y_mask = x.to(self.device), attn_mask.to(self.device), y_mask.to(self.device)

                with torch.no_grad():
                    y_predict = self.model(x, attn_mask)
                    y_predict = y_predict.view(-1, y_predict.shape[2])
                    y_predict = torch.argmax(y_predict, dim=1).view(-1)
                for i in range(y_mask.shape[0]):
                    if y_mask[i] == 1:
                        result += words[decode_idx] + \
                            self.targets[y_predict[i].item()] + ' '
                        decode_idx += 1
        result = result.strip()
        return result
