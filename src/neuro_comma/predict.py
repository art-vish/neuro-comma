from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Type, Union

import torch
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from neuro_comma.dataset import BaseDataset
from neuro_comma.model import CorrectionModel
from neuro_comma.pretrained import PRETRAINED_MODELS
from neuro_comma.utils import get_last_pretrained_weight_path, load_params


class BasePredictor:
    def __init__(self,
                 targets: dict[str, int],
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
        
        self.device = torch.device('cuda' if (not quantization) and torch.cuda.is_available() else 'cpu')
        self.targets = targets
        self.pretrained_model = pretrained_model
        self.pretrained_model_path = str(pretrained_model_path) if pretrained_model_path else pretrained_model
        self.pretrained_tokenizer_model_path = str(pretrained_tokenizer_model_path) if pretrained_tokenizer_model_path else pretrained_model
        self.weights = model_weights_path
        self.freeze_pretrained = freeze_pretrained
        self.lstm_dim = lstm_dim
        self.sequence_length = sequence_length

        self.model = self.load_model(quantization=quantization)
        self.tokenizer = self.load_tokenizer()
        self.dataset_class = dataset_class

    def load_model(self, quantization: Optional[bool] = False) -> CorrectionModel:
        model = CorrectionModel(self.pretrained_model,
                                self.pretrained_model_path,
                                self.targets,
                                self.freeze_pretrained,
                                self.lstm_dim)

        if quantization:
            model = model.quantize()

        model.to(self.device)
        model.load(self.weights, map_location=self.device)
        model.eval()
        return model

    def load_tokenizer(self) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:        
        tokenizer = PRETRAINED_MODELS[self.pretrained_model][1].from_pretrained(self.pretrained_tokenizer_model_path)
        return tokenizer


class RepunctPredictor(BasePredictor):
    def __call__(self, text: str, decode_map: Dict[int, str] = {0: '', 1: ',', 2: '.', 3: '?'}) -> str:
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
        x = torch.index_select(data, 1, x_indecies).reshape(2, -1).to(self.device)

        attn_mask_indecies = torch.tensor([2])
        attn_mask = torch.index_select(data, 1, attn_mask_indecies).reshape(2, -1).to(self.device)

        y_indecies = torch.tensor([4])
        y_mask = torch.index_select(data, 1, y_indecies).view(-1)

        with torch.no_grad():
            y_predict = self.model(x, attn_mask)

        y_predict = y_predict.view(-1, y_predict.shape[2])
        y_predict = torch.argmax(y_predict, dim=1).view(-1)

        for i in range(y_mask.shape[0]):
            if y_mask[i] == 1:
                result += words_original_case[decode_idx]
                result += decode_map[y_predict[i].item()]
                result += ' '
                decode_idx += 1

        result = result.strip()
        return result
