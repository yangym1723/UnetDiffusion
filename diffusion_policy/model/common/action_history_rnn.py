from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


RNNState = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


class ActionHistoryRNN(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: Optional[int] = None,
            rnn_type: str = 'gru',
            num_layers: int = 1,
            dropout: float = 0.0,
            use_cumulative_sum: bool = True,
            output_mlp_dims: Optional[Sequence[int]] = None,
            nonlinearity: str = 'tanh'):
        super().__init__()
        if output_dim is None:
            output_dim = hidden_dim
        if output_mlp_dims is None:
            output_mlp_dims = []

        rnn_type = rnn_type.lower()
        rnn_dropout = dropout if num_layers > 1 else 0.0
        if rnn_type == 'gru':
            rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=rnn_dropout,
                batch_first=True)
        elif rnn_type == 'lstm':
            rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=rnn_dropout,
                batch_first=True)
        elif rnn_type == 'rnn':
            rnn = nn.RNN(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=rnn_dropout,
                nonlinearity=nonlinearity,
                batch_first=True)
        else:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}")

        projection_layers = []
        last_dim = hidden_dim
        for dim in output_mlp_dims:
            projection_layers.append(nn.Linear(last_dim, dim))
            projection_layers.append(nn.Mish())
            last_dim = dim
        if last_dim != output_dim:
            projection_layers.append(nn.Linear(last_dim, output_dim))
        self.output_proj = nn.Identity() if len(projection_layers) == 0 else nn.Sequential(*projection_layers)

        self.rnn = rnn
        self.rnn_type = rnn_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.use_cumulative_sum = use_cumulative_sum

    @property
    def is_lstm(self) -> bool:
        return self.rnn_type == 'lstm'

    def get_zero_state(self, batch_size: int, device, dtype) -> RNNState:
        state = torch.zeros(
            (self.num_layers, batch_size, self.hidden_dim),
            device=device,
            dtype=dtype)
        if self.is_lstm:
            return (state, state.clone())
        return state

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_proj(x)

    def forward(self, action_history: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        action_history: (B, L, Da), padded with zeros.
        lengths: (B,), valid prefix length for each sequence.
        """
        batch_size, max_length, _ = action_history.shape
        if max_length == 0:
            return torch.zeros(
                (batch_size, self.output_dim),
                device=action_history.device,
                dtype=action_history.dtype)
        feature = torch.zeros(
            (batch_size, self.output_dim),
            device=action_history.device,
            dtype=action_history.dtype)
        valid_mask = lengths > 0
        if not valid_mask.any():
            return feature

        valid_history = action_history[valid_mask]
        valid_lengths = lengths[valid_mask].to(dtype=torch.long)
        max_valid_length = int(valid_lengths.max().item())
        valid_history = valid_history[:, :max_valid_length]

        rnn_input = valid_history
        if self.use_cumulative_sum:
            rnn_input = torch.cumsum(valid_history, dim=1)

        packed_input = pack_padded_sequence(
            rnn_input,
            lengths=valid_lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False)
        initial_state = self.get_zero_state(
            batch_size=valid_history.shape[0],
            device=action_history.device,
            dtype=action_history.dtype)
        _, final_state = self.rnn(packed_input, initial_state)
        if self.is_lstm:
            last_hidden = final_state[0][-1]
        else:
            last_hidden = final_state[-1]
        projected = self._project(last_hidden)
        if projected.dtype != feature.dtype:
            projected = projected.to(feature.dtype)
        feature[valid_mask] = projected
        return feature

    def step(self, cumulative_action: torch.Tensor, prev_state: Optional[RNNState]) -> Tuple[torch.Tensor, RNNState]:
        """
        cumulative_action: (B, Da), cumulative sum up to the current executed step.
        """
        if prev_state is None:
            prev_state = self.get_zero_state(
                batch_size=cumulative_action.shape[0],
                device=cumulative_action.device,
                dtype=cumulative_action.dtype)
        output, next_state = self.rnn(cumulative_action[:, None, :], prev_state)
        feature = self._project(output[:, -1])
        return feature, next_state
