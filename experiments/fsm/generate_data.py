"""
Finite State Machine (FSM) data generation.

Creates synthetic data from a finite state machine where:
- Each state has a set of valid output tokens
- Transitions depend on the current state and emitted token
- The model must learn to track state to predict valid next tokens

This replicates the setup from the Stanford "Codebook Features" paper,
which showed that VQ bottlenecks learn to represent FSM states.
"""

import random
from dataclasses import dataclass

import torch


@dataclass
class FSMConfig:
    """Configuration for FSM generation."""
    num_states: int = 10
    tokens_per_state: int = 3
    vocab_size: int = 30  # Should be >= num_states * tokens_per_state
    seed: int = 42


class FiniteStateMachine:
    """
    A finite state machine for sequence generation.

    Each state emits tokens from a disjoint set (no token overlap between states).
    Transitions are deterministic based on (state, token) pairs.
    """

    def __init__(self, config: FSMConfig):
        self.config = config
        self.num_states = config.num_states
        self.tokens_per_state = config.tokens_per_state
        self.vocab_size = config.vocab_size

        random.seed(config.seed)

        # Assign disjoint token sets to each state
        # State i emits tokens [i*tokens_per_state, (i+1)*tokens_per_state)
        self.state_tokens = {}
        for state in range(self.num_states):
            start = state * self.tokens_per_state
            self.state_tokens[state] = list(range(start, start + self.tokens_per_state))

        # Create transition table: (state, token) -> next_state
        # Random but deterministic transitions
        self.transitions = {}
        for state in range(self.num_states):
            for token in self.state_tokens[state]:
                next_state = random.randint(0, self.num_states - 1)
                self.transitions[(state, token)] = next_state

        # Reverse lookup: token -> state that emits it
        self.token_to_state = {}
        for state, tokens in self.state_tokens.items():
            for token in tokens:
                self.token_to_state[token] = state

    def get_state_for_token(self, token: int) -> int:
        """Return the state that emits this token."""
        return self.token_to_state[token]

    def get_valid_tokens(self, state: int) -> list[int]:
        """Return tokens that can be emitted from this state."""
        return self.state_tokens[state]

    def step(self, state: int, token: int) -> int:
        """Return next state given current state and emitted token."""
        return self.transitions[(state, token)]

    def generate_sequence(self, length: int, start_state: int = 0) -> tuple[list[int], list[int]]:
        """
        Generate a sequence of tokens and corresponding states.

        Returns:
            tokens: List of emitted tokens
            states: List of states (state[i] is the state that emitted token[i])
        """
        tokens = []
        states = []
        state = start_state

        for _ in range(length):
            # Record current state
            states.append(state)

            # Emit random token from current state
            token = random.choice(self.state_tokens[state])
            tokens.append(token)

            # Transition to next state
            state = self.transitions[(state, token)]

        return tokens, states

    def generate_batch(
        self,
        batch_size: int,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate a batch of sequences.

        Returns:
            inputs: (batch_size, seq_len) - input tokens
            targets: (batch_size, seq_len) - target tokens (shifted by 1)
            states: (batch_size, seq_len) - ground truth states
        """
        all_tokens = []
        all_states = []

        for _ in range(batch_size):
            start_state = random.randint(0, self.num_states - 1)
            tokens, states = self.generate_sequence(seq_len + 1, start_state)
            all_tokens.append(tokens)
            all_states.append(states)

        # Convert to tensors
        tokens_tensor = torch.tensor(all_tokens, dtype=torch.long)
        states_tensor = torch.tensor(all_states, dtype=torch.long)

        # Inputs are tokens[:-1], targets are tokens[1:]
        inputs = tokens_tensor[:, :-1]
        targets = tokens_tensor[:, 1:]
        states = states_tensor[:, :-1]  # States corresponding to inputs

        return inputs, targets, states


def compute_state_code_alignment(
    hard_codes: torch.Tensor,
    states: torch.Tensor,
    num_states: int,
) -> dict:
    """
    Compute how well codes align with FSM states.

    For each code, compute which state it most frequently co-occurs with.
    Returns metrics on the quality of this alignment.

    Args:
        hard_codes: (batch, seq_len, codebook_size) - active codes
        states: (batch, seq_len) - ground truth states

    Returns:
        Dictionary with alignment metrics
    """
    batch, seq_len, codebook_size = hard_codes.shape

    # Flatten for analysis
    codes_flat = hard_codes.reshape(-1, codebook_size)  # (B*S, C)
    states_flat = states.reshape(-1)  # (B*S,)

    # For each code, count co-occurrences with each state
    # code_state_counts[c, s] = how often code c is active when in state s
    code_state_counts = torch.zeros(codebook_size, num_states)

    for c in range(codebook_size):
        code_active = codes_flat[:, c] > 0.5  # Positions where code c is active
        if code_active.sum() > 0:
            for s in range(num_states):
                state_mask = states_flat == s
                code_state_counts[c, s] = (code_active & state_mask).sum().item()

    # For each code, find its dominant state
    dominant_states = code_state_counts.argmax(dim=1)
    max_counts = code_state_counts.max(dim=1).values
    total_counts = code_state_counts.sum(dim=1)

    # Purity: fraction of code activations that match dominant state
    purity = (max_counts / (total_counts + 1e-8)).mean().item()

    # Coverage: how many codes are assigned to each state
    codes_per_state = torch.zeros(num_states)
    for s in range(num_states):
        codes_per_state[s] = (dominant_states == s).sum()

    # Active codes (used at least once)
    active_codes = (total_counts > 0).sum().item()

    return {
        'purity': purity,  # Higher = codes specialize to states
        'active_codes': active_codes,
        'codes_per_state_mean': codes_per_state.mean().item(),
        'codes_per_state_std': codes_per_state.std().item(),
        'code_state_counts': code_state_counts,
        'dominant_states': dominant_states,
    }


if __name__ == "__main__":
    # Demo: generate and print FSM data
    config = FSMConfig(num_states=5, tokens_per_state=3, vocab_size=15)
    fsm = FiniteStateMachine(config)

    print("FSM Configuration:")
    print(f"  States: {fsm.num_states}")
    print(f"  Tokens per state: {fsm.tokens_per_state}")
    print(f"  Vocab size: {fsm.vocab_size}")
    print()

    print("State -> Tokens mapping:")
    for state, tokens in fsm.state_tokens.items():
        print(f"  State {state}: tokens {tokens}")
    print()

    print("Sample transitions:")
    for (state, token), next_state in list(fsm.transitions.items())[:10]:
        print(f"  ({state}, {token}) -> {next_state}")
    print()

    print("Sample sequence:")
    tokens, states = fsm.generate_sequence(20)
    print(f"  Tokens: {tokens}")
    print(f"  States: {states}")
