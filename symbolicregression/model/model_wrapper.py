# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch
import torch.nn as nn


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class ModelWrapper(nn.Module):
    """"""

    def __init__(
        self,
        env=None,
        embedder=None,
        encoder=None,
        decoder=None,
        beam_type="search",
        beam_length_penalty=1,
        beam_size=1,
        beam_early_stopping=True,
        max_generated_output_len=200,
        beam_temperature=1.0,
    ):
        super().__init__()

        self.env = env
        self.embedder = embedder
        self.encoder = encoder
        self.decoder = decoder
        self.beam_type = beam_type
        self.beam_early_stopping = beam_early_stopping
        self.max_generated_output_len = max_generated_output_len
        self.beam_size = beam_size
        self.beam_length_penalty = beam_length_penalty
        self.beam_temperature = beam_temperature
        self.device = next(self.embedder.parameters()).device

    @torch.no_grad()
    def init_state(
        self, state,
    ):
        #print("MIDDLE init_state")
        env = self.env
        embedder, encoder, decoder = self.embedder, self.encoder, self.decoder

        x, x_len = embedder(state.inputs)
        encoded = encoder("fwd", x=x, lengths=x_len, causal=False).transpose(0, 1)
        state.encoded = encoded
        state.x_len = x_len
        state = decoder.init_state(state)
        return state

    @torch.no_grad()
    def get_policy(
        self, state,
    ):

        """
        x: bags of sequences (B, T)
        """
        # print("MIDDLE POLICY")
        env = self.env
        embedder, encoder, decoder = self.embedder, self.encoder, self.decoder

        ### Greedy solution.
        # print(type(decoder))
        policy = decoder.get_policy(
            state,
            sample_temperature=None,
            max_len=self.max_generated_output_len,
        )

        return policy

    @torch.no_grad()
    def apply_action(
        self, state, idx
    ):

        """
        x: bags of sequences (B, T)
        """
        # print("MIDDLE apply_action")
        env = self.env
        embedder, encoder, decoder = self.embedder, self.encoder, self.decoder

        ### Greedy solution.
        # print(type(decoder))
        state = decoder.apply_action(
            state,
            sample_temperature=None,
            max_len=self.max_generated_output_len,
            idx=idx
        )

        return state
    
    def is_solution(self, state):
        env = self.env
        embedder, encoder, decoder = self.embedder, self.encoder, self.decoder

        return decoder.is_solution(state, self.max_generated_output_len)

    @torch.no_grad()
    def sample_candidates_from_state(
        self, state,
    ):
        env = self.env
        embedder, encoder, decoder = self.embedder, self.encoder, self.decoder

        self.decoder.cleanup(state)
        generations = state.generated
        generations = generations.unsqueeze(-1).view(generations.shape[0], state.bs, 1)
        generations = generations.transpose(0, 1).transpose(1, 2).cpu().tolist()
        generations = [
            list(
                filter(
                    lambda x: x is not None,
                    [
                        env.idx_to_infix(hyp[1:-1], is_float=False, str_array=False)
                        for hyp in generations[i]
                    ],
                )
            )
            for i in range(state.bs)
        ]

        outputs = []

        print("Sampling")
        num_samples = self.beam_size
        encoded = (
            state.encoded.unsqueeze(1)
            .expand((state.bs, num_samples) + state.encoded.shape[1:])
            .contiguous()
            .view((state.bs * num_samples,) + state.encoded.shape[1:])
        )
        x_len = state.x_len.unsqueeze(1).expand(state.bs, num_samples).contiguous().view(-1)
        sampling_generations, _ = decoder.generate(
            encoded,
            x_len,
            sample_temperature=self.beam_temperature,
            max_len=self.max_generated_output_len,
        )
        sampling_generations = sampling_generations.unsqueeze(-1).view(
            sampling_generations.shape[0], state.bs, num_samples
        )
        sampling_generations = (
            sampling_generations.transpose(0, 1).transpose(1, 2).cpu().tolist()
        )
        sampling_generations = [
            list(
                filter(
                    lambda x: x is not None,
                    [
                        env.idx_to_infix(
                            hyp[1:-1], is_float=False, str_array=False
                        )
                        for hyp in sampling_generations[i]
                    ],
                )
            )
            for i in range(state.bs)
        ]
        for i in range(state.bs):
            generations[i].extend(sampling_generations[i])

        outputs.extend(generations)

        print("MODEL_WRAPPER sample_candidates_from_state")
        #for g in outputs:
        #    print(outputs)
        #print("MODEL_WRAPPER sample_candidates_from_state")
        return outputs

    @torch.no_grad()
    def forward(
        self, input,
    ):

        """
        x: bags of sequences (B, T)
        """

        env = self.env
        embedder, encoder, decoder = self.embedder, self.encoder, self.decoder

        B, T = len(input), max([len(xi) for xi in input])
        outputs = []

        for chunk in chunks(
            np.arange(B),
            min(
                int(10000 / T),
                int(100000 / self.beam_size / self.max_generated_output_len),
            ),
        ):
            x, x_len = embedder([input[idx] for idx in chunk])
            #print(f"x_len: {x_len}")
            encoded = encoder("fwd", x=x, lengths=x_len, causal=False).transpose(0, 1)
            bs = encoded.shape[0]

            ### Greedy solution.
            #print(f"Generation x_len: {x_len}")
            generations, _ = decoder.generate(
                encoded,
                x_len,
                sample_temperature=None,
                max_len=self.max_generated_output_len,
            )

            generations = generations.unsqueeze(-1).view(generations.shape[0], bs, 1)
            generations = generations.transpose(0, 1).transpose(1, 2).cpu().tolist()
            generations = [
                list(
                    filter(
                        lambda x: x is not None,
                        [
                            env.idx_to_infix(hyp[1:-1], is_float=False, str_array=False)
                            for hyp in generations[i]
                        ],
                    )
                )
                for i in range(bs)
            ]
            print("MODEL_WRAPPER generations")
            #for g in generations:
            #    print(g)
            #print("MODEL_WRAPPER generations")
            # return [g]

            # @NOTE: JAKE
            # I think this is the refinement step in the paper (2.2)
            # or maybe sampling step before refinement
            if self.beam_type == "search":
                print("MODEL WRAPPER search")
                _, _, search_generations = decoder.generate_beam(
                    encoded,
                    x_len,
                    beam_size=self.beam_size,
                    length_penalty=self.beam_length_penalty,
                    max_len=self.max_generated_output_len,
                    early_stopping=self.beam_early_stopping,
                )
                search_generations = [
                    sorted(
                        [hyp for hyp in search_generations[i].hyp],
                        key=lambda s: s[0],
                        reverse=True,
                    )
                    for i in range(bs)
                ]
                search_generations = [
                    list(
                        filter(
                            lambda x: x is not None,
                            [
                                env.idx_to_infix(
                                    hyp.cpu().tolist()[1:],
                                    is_float=False,
                                    str_array=False,
                                )
                                for (_, hyp) in search_generations[i]
                            ],
                        )
                    )
                    for i in range(bs)
                ]
                for i in range(bs):
                    generations[i].extend(search_generations[i])

            elif self.beam_type == "sampling":
                print("MODEL WRAPPER sampling")
                num_samples = self.beam_size
                encoded = (
                    encoded.unsqueeze(1)
                    .expand((bs, num_samples) + encoded.shape[1:])
                    .contiguous()
                    .view((bs * num_samples,) + encoded.shape[1:])
                )
                #print(f"sampling x_len: {x_len}")
                x_len = x_len.unsqueeze(1).expand(bs, num_samples).contiguous().view(-1)
                #print(f"sampling x_len: {x_len}")
                sampling_generations, _ = decoder.generate(
                    encoded,
                    x_len,
                    sample_temperature=self.beam_temperature,
                    max_len=self.max_generated_output_len,
                )
                sampling_generations = sampling_generations.unsqueeze(-1).view(
                    sampling_generations.shape[0], bs, num_samples
                )
                sampling_generations = (
                    sampling_generations.transpose(0, 1).transpose(1, 2).cpu().tolist()
                )
                sampling_generations = [
                    list(
                        filter(
                            lambda x: x is not None,
                            [
                                env.idx_to_infix(
                                    hyp[1:-1], is_float=False, str_array=False
                                )
                                for hyp in sampling_generations[i]
                            ],
                        )
                    )
                    for i in range(bs)
                ]
                for i in range(bs):
                    generations[i].extend(sampling_generations[i])
            else:
                raise NotImplementedError
            outputs.extend(generations)
        # for chunk in chunks

        print("MODEL_WRAPPER generations")
        #for g in outputs:
        #    print(outputs)
        #print("MODEL_WRAPPER generations")
        return outputs
