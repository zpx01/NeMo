# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import torch

from nemo.collections.asr.parts.utils import rnnt_utils
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.core.classes import Typing, typecheck
from nemo.core.neural_types import HypothesisType, LengthsType, LogprobsType, NeuralType
from nemo.utils import logging

DEFAULT_TOKEN_OFFSET = 100


def pack_hypotheses(
    hypotheses: List[rnnt_utils.NBestHypotheses], logitlen: torch.Tensor,
) -> List[rnnt_utils.NBestHypotheses]:

    if logitlen is not None:
        if hasattr(logitlen, 'cpu'):
            logitlen_cpu = logitlen.to('cpu')
        else:
            logitlen_cpu = logitlen

    for idx, hyp in enumerate(hypotheses):  # type: rnnt_utils.NBestHypotheses
        for candidate_idx, cand in enumerate(hyp.n_best_hypotheses):
            cand.y_sequence = torch.tensor(cand.y_sequence, dtype=torch.long)

            if logitlen is not None:
                cand.length = logitlen_cpu[idx]

            if cand.dec_state is not None:
                cand.dec_state = _states_to_device(cand.dec_state)

    return hypotheses


def _states_to_device(dec_state, device='cpu'):
    if torch.is_tensor(dec_state):
        dec_state = dec_state.to(device)

    elif isinstance(dec_state, (list, tuple)):
        dec_state = tuple(_states_to_device(dec_i, device) for dec_i in dec_state)

    return dec_state


class AbstractBeamCTCInfer(Typing):
    """A beam CTC decoder.

    Provides a common abstraction for sample level beam decoding.

    Args:
        blank_id: int, index of the blank token. Can be 0 or len(vocabulary).
        beam_size: int, size of the beam used in the underlying beam search engine.

    """

    @property
    def input_types(self):
        """Returns definitions of module input ports.
        """
        return {
            "decoder_output": NeuralType(('B', 'T', 'D'), LogprobsType()),
            "decoder_lengths": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        """Returns definitions of module output ports.
        """
        return {"predictions": [NeuralType(elements_type=HypothesisType())]}

    def __init__(self, blank_id: int, beam_size: int):
        self.blank_id = blank_id

        if beam_size < 1:
            raise ValueError("Beam search size cannot be less than 1!")

        self.beam_size = beam_size

        # Variables set by corresponding setter methods
        self.vocab = None
        self.decoding_type = None
        self.tokenizer = None

        # Utility maps for vocabulary
        self.vocab_index_map = None
        self.index_vocab_map = None

        # Internal variable, used to prevent double reduction of consecutive tokens (ctc collapse)
        self.override_fold_consecutive_value = None

    def set_vocabulary(self, vocab: List[str]):
        """
        Set the vocabulary of the decoding framework.

        Args:
            vocab: List of str. Each token corresponds to its location in the vocabulary emitted by the model.
                Note that this vocabulary must NOT contain the "BLANK" token.
        """
        self.vocab = vocab
        self.vocab_index_map = {v: i for i, v in enumerate(vocab)}
        self.index_vocab_map = {i: v for i, v in enumerate(vocab)}

    def set_decoding_type(self, decoding_type: str):
        """
        Sets the decoding type of the framework. Can support either char or subword models.

        Args:
            decoding_type: Str corresponding to decoding type. Only supports "char" and "subword".
        """
        decoding_type = decoding_type.lower()
        supported_types = ['char', 'subword']

        if decoding_type not in supported_types:
            raise ValueError(
                f"Unsupported decoding type. Supported types = {supported_types}.\n" f"Given = {decoding_type}"
            )

        self.decoding_type = decoding_type

    def set_tokenizer(self, tokenizer: TokenizerSpec):
        """
        Set the tokenizer of the decoding framework.

        Args:
            tokenizer: NeMo tokenizer object, which inherits from TokenizerSpec.
        """
        self.tokenizer = tokenizer

    @typecheck()
    def forward(
        self, decoder_output: torch.Tensor, decoder_lengths: torch.Tensor,
    ) -> Tuple[List[Union[rnnt_utils.Hypothesis, rnnt_utils.NBestHypotheses]]]:
        """Returns a list of hypotheses given an input batch of the encoder hidden embedding.
        Output token is generated auto-repressively.

        Args:
            decoder_output: A tensor of size (batch, timesteps, features) or (batch, timesteps) (each timestep is a label).
            decoder_lengths: list of int representing the length of each sequence
                output sequence.

        Returns:
            packed list containing batch number of sentences (Hypotheses).
        """
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class BeamCTCInfer(AbstractBeamCTCInfer):
    """A greedy CTC decoder.

    Provides a common abstraction for sample level and batch level greedy decoding.

    Args:
        blank_index: int index of the blank token. Can be 0 or len(vocabulary).
        preserve_alignments: Bool flag which preserves the history of logprobs generated during
            decoding (sample / batched). When set to true, the Hypothesis will contain
            the non-null value for `logprobs` in it. Here, `logprobs` is a torch.Tensors.
        compute_timestamps: A bool flag, which determines whether to compute the character/subword, or
                word based timestamp mapping the output log-probabilities to discrite intervals of timestamps.
                The timestamps will be available in the returned Hypothesis.timestep as a dictionary.

    """

    def __init__(
        self,
        blank_id: int,
        beam_size: int,
        return_best_hypothesis: bool = True,
        preserve_alignments: bool = False,
        compute_timestamps: bool = False,
        beam_alpha: float = 1.0,
        beam_beta: float = 0.0,
        kenlm_path: str = None,
        flashlight_cfg: Optional['FlashlightConfig'] = None,
        pyctcdecode_cfg: Optional['PyCTCDecodeConfig'] = None,
    ):
        super().__init__(blank_id=blank_id, beam_size=beam_size)

        self.return_best_hypothesis = return_best_hypothesis
        self.preserve_alignments = preserve_alignments
        self.compute_timestamps = compute_timestamps

        if self.compute_timestamps:
            raise ValueError(f"Currently this flag is not supported for beam search algorithms.")

        self.vocab = None  # This must be set by specific method by user before calling forward() !
        self.search_algorithm = self.flashlight_beam_search

        self.beam_alpha = beam_alpha
        self.beam_beta = beam_beta

        # Default beam search args
        self.kenlm_path = kenlm_path

        if flashlight_cfg is None:
            flashlight_cfg = FlashlightConfig()
        self.flashlight_cfg = flashlight_cfg

        # Default beam search scorer functions
        self.default_beam_scorer = None
        self.pyctcdecode_beam_scorer = None
        self.flashlight_beam_scorer = None
        self.token_offset = 0

    @typecheck()
    def forward(
        self, decoder_output: torch.Tensor, decoder_lengths: torch.Tensor,
    ) -> Tuple[List[Union[rnnt_utils.Hypothesis, rnnt_utils.NBestHypotheses]]]:
        """Returns a list of hypotheses given an input batch of the encoder hidden embedding.
        Output token is generated auto-repressively.

        Args:
            decoder_output: A tensor of size (batch, timesteps, features).
            decoder_lengths: list of int representing the length of each sequence
                output sequence.

        Returns:
            packed list containing batch number of sentences (Hypotheses).
        """
        if self.vocab is None:
            raise RuntimeError("Please set the vocabulary with `set_vocabulary()` before calling this function.")

        if self.decoding_type is None:
            raise ValueError("Please set the decoding type with `set_decoding_type()` before calling this function.")

        with torch.no_grad(), torch.inference_mode():
            # Process each sequence independently
            prediction_tensor = decoder_output

            if prediction_tensor.ndim != 3:
                raise ValueError(
                    f"`decoder_output` must be a tensor of shape [B, T, V] (log probs, float). "
                    f"Provided shape = {prediction_tensor.shape}"
                )

            # determine type of input - logprobs or labels
            out_len = decoder_lengths if decoder_lengths is not None else None
            hypotheses = self.search_algorithm(prediction_tensor, out_len)

            # Pack results into Hypotheses
            packed_result = pack_hypotheses(hypotheses, decoder_lengths)

            # Pack the result
            if self.return_best_hypothesis and isinstance(packed_result[0], rnnt_utils.NBestHypotheses):
                packed_result = [res.n_best_hypotheses[0] for res in packed_result]  # type: Hypothesis

        return (packed_result,)


    @torch.no_grad()
    def flashlight_beam_search(
        self, x: torch.Tensor, out_len: torch.Tensor
    ) -> List[Union[rnnt_utils.Hypothesis, rnnt_utils.NBestHypotheses]]:
        """
        Flashlight Beam Search Algorithm. Should support Char and Subword models.

        Args:
            x: Tensor of shape [B, T, V+1], where B is the batch size, T is the maximum sequence length,
                and V is the vocabulary size. The tensor contains log-probabilities.
            out_len: Tensor of shape [B], contains lengths of each sequence in the batch.

        Returns:
            A list of NBestHypotheses objects, one for each sequence in the batch.
        """
        if self.compute_timestamps:
            raise ValueError(
                f"Flashlight beam search does not support time stamp calculation!"
            )

        if self.flashlight_beam_scorer is None:
            # Check for filepath
            if self.kenlm_path is None or not os.path.exists(self.kenlm_path):
                raise FileNotFoundError(
                    f"KenLM binary file not found at : {self.kenlm_path}. "
                    f"Please set a valid path in the decoding config."
                )

            # perform token offset for subword models
            # if self.decoding_type == 'subword':
            #    vocab = [chr(idx + self.token_offset) for idx in range(len(self.vocab))]
            # else:
            #    # char models
            #    vocab = self.vocab

            # Must import at runtime to avoid circular dependency due to module level import.
            from nemo.collections.asr.modules.flashlight_decoder import FlashLightKenLMBeamSearchDecoder

            self.flashlight_beam_scorer = FlashLightKenLMBeamSearchDecoder(
                lm_path=self.kenlm_path,
                vocabulary=self.vocab,
                tokenizer=self.tokenizer,
                lexicon_path=self.flashlight_cfg.lexicon_path,
                boost_path=self.flashlight_cfg.boost_path,
                beam_size=self.beam_size,
                beam_size_token=self.flashlight_cfg.beam_size_token,
                beam_threshold=self.flashlight_cfg.beam_threshold,
                lm_weight=self.beam_alpha,
                word_score=self.beam_beta,
                unk_weight=self.flashlight_cfg.unk_weight,
                sil_weight=self.flashlight_cfg.sil_weight,
            )

        x = x.to('cpu')

        with typecheck.disable_checks():
            beams_batch = self.flashlight_beam_scorer.forward(log_probs=x)

        # For each sample in the batch
        nbest_hypotheses = []
        for beams_idx, beams in enumerate(beams_batch):
            # For each beam candidate / hypothesis in each sample
            hypotheses = []
            for candidate_idx, candidate in enumerate(beams):
                hypothesis = rnnt_utils.Hypothesis(
                    score=0.0, y_sequence=[], dec_state=None, timestep=[], last_token=None
                )

                # We preserve the token ids and the score for this hypothesis
                hypothesis.y_sequence = candidate['tokens'].tolist()
                hypothesis.score = candidate['score']

                # If alignment must be preserved, we preserve a view of the output logprobs.
                # Note this view is shared amongst all beams within the sample, be sure to clone it if you
                # require specific processing for each sample in the beam.
                # This is done to preserve memory.
                if self.preserve_alignments:
                    hypothesis.alignments = x[beams_idx][: out_len[beams_idx]]

                hypotheses.append(hypothesis)

            # Wrap the result in NBestHypothesis.
            hypotheses = rnnt_utils.NBestHypotheses(hypotheses)
            nbest_hypotheses.append(hypotheses)

        return nbest_hypotheses

    def set_decoding_type(self, decoding_type: str):
        super().set_decoding_type(decoding_type)

        # Please check train_kenlm.py in scripts/asr_language_modeling/ to find out why we need
        # TOKEN_OFFSET for BPE-based models
        if self.decoding_type == 'subword':
            self.token_offset = DEFAULT_TOKEN_OFFSET


@dataclass
class FlashlightConfig:
    lexicon_path: Optional[str] = None
    boost_path: Optional[str] = None
    beam_size_token: int = 16
    beam_threshold: float = 20.0
    unk_weight: float = -math.inf
    sil_weight: float = 0.0


@dataclass
class BeamCTCInferConfig:
    beam_size: int
    preserve_alignments: bool = False
    compute_timestamps: bool = False
    return_best_hypothesis: bool = True

    beam_alpha: float = 1.0
    beam_beta: float = 0.0
    kenlm_path: Optional[str] = None

    flashlight_cfg: Optional[FlashlightConfig] = field(default_factory=lambda: FlashlightConfig())
    
@dataclass
class BeamCTCInferConfigList(BeamCTCInferConfig):
    beam_size: List[int]
    beam_alpha: List[float]
    beam_beta: List[float]
