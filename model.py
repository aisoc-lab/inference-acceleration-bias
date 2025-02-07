from typing import Tuple, Optional, List, Dict, Any
from transformers import (
    GenerationConfig,
)
import torch
import torch.nn as nn
import bitsandbytes as bnb
import logging
from awq import AutoAWQForCausalLM

from model_loader import ModelLoader

logger = logging.getLogger(__name__)


class Model:
    """
    Base class for all models. Represents a model under evaluation.

    Handles everything related to the model.
    """

    def __init__(self, args):
        self.args = args
        self.awq = False

        deployment = self.args.deployment if "deployment" in self.args else None
        (
            self.model,
            self.tokenizer,
            generation_config,
            self.cache_implementation,
            self.cache_config,
        ) = ModelLoader.load_model(
            self.args.model_name, deployment, self.args.sampling_method
        )

        self.awq = isinstance(self.model, AutoAWQForCausalLM)

        logger.info(f"model loaded: {self.args.model_name}")

        # Set the generation config
        self.model.generation_config = generation_config

    def predict(self, prompt: str) -> Tuple[Optional[str], Optional[float]]:
        """
        Runs the model on the given prompt, by prompting the model with the prompt, and extracting the output and log probabilities from the model.

        Args:
            prompt: The prompt to run the model on.

        Returns:
            Tuple containing the output and log probabilities from the model.

        """
        # compose payload for the model from the prompt TODO
        # payload = self.compose_payload(prompt)
        # run the model on the payload
        with torch.inference_mode():
            inputs = self.tokenizer([prompt], return_tensors="pt").input_ids.to("cuda")
            outputs = self.model(inputs, labels=inputs)
        log_probability = (
            -outputs.loss.item()
        )  # log probability is the negative loss (negative log likelihood)
        return outputs, log_probability

    def generate(self, prompt: str) -> str:
        """
        Generates text from the model on a prompt.

        Args:
            prompt: The prompt to generate text from.

        Returns:
            The generated text.

        """
        try:
            with torch.no_grad():
                inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

                # Generate text with the model
                generations = self.model.generate(
                    **inputs,
                    max_new_tokens=self.args.max_new_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                    generation_config=self.model.generation_config,
                    cache_implementation=self.cache_implementation,
                    cache_config=self.cache_config,
                )

                # Check if the generated text contains the input prompt
                gen_contains_input = (
                    inputs["input_ids"][0] == generations[0][: inputs["input_ids"].shape[1]]  # type: ignore
                ).all()

                # Handle prompt removal if required
                if (
                    self.args.remove_prompt_from_generated_text
                    and not gen_contains_input
                ):
                    logger.warn(
                        "Warning: The generated text does not contain the input prompt."
                    )

                if gen_contains_input and self.args.remove_prompt_from_generated_text:
                    text = self.tokenizer.batch_decode(
                        generations[:, inputs["input_ids"].shape[1] :]  # type: ignore
                    )[0]
                else:
                    text = self.tokenizer.batch_decode(
                        generations, skip_special_tokens=True
                    )[0]

                # Free up memory
                del generations
                torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            logger.error("Failed again after clearing cache. Skipping this prompt.")

        except Exception as e:
            logger.error(f"An error occurred: {e}")

        return text

    def generate_mc(
        self,
        prompt: str,
        num_options: int,
        mc_type: str = "alpha",
    ) -> Tuple[List[Dict[int, float]], List[float]]:
        """
        Generates text from the model on a list of prompts.

        Args:
            prompts: A tuple of a prompt and the number of answer options ([str], [int]).
            mc_type: The type of multiple choice options, 'alpha' for A, B, C, (depends on the number of options), 'yn' for Yes, No.

        Returns:
            List of dicts, each dict contains the probabilities of each option.
        """

        answer_dict = {}

        try:
            with torch.no_grad():
                inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

                output = self.model.generate(
                    **inputs,
                    max_new_tokens=self.args.max_new_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                    generation_config=self.model.generation_config,
                    cache_implementation=self.cache_implementation,
                    cache_config=self.cache_config,
                )

                pd = output.scores[0][0].softmax(dim=0)

                if mc_type == "alpha":
                    for i in range(num_options):
                        tok_id = self.tokenizer.convert_tokens_to_ids(f"{chr(65+i)}")
                        answer_dict[i] = pd[tok_id].item()
                elif mc_type == "yn":
                    assert num_options == 2
                    answer_dict[0] = pd[
                        self.tokenizer.convert_tokens_to_ids("yes")
                    ].item()
                    answer_dict[1] = pd[
                        self.tokenizer.convert_tokens_to_ids("no")
                    ].item()

                sum_pd = sum(answer_dict.values())
                for i in range(num_options):
                    answer_dict[i] /= sum_pd

                p_mass = 1 - sum_pd

                del output
                torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA out of memory. Clearing cache and retrying.")
            torch.cuda.empty_cache()
            try:
                with torch.no_grad():
                    inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=self.args.max_new_tokens,
                        pad_token_id=self.tokenizer.eos_token_id,
                        return_dict_in_generate=True,
                        output_scores=True,
                        generation_config=self.model.generation_config,
                        cache_implementation=self.cache_implementation,
                        cache_config=self.cache_config,
                    )

                    pd = output.scores[0][0].softmax(dim=0)

                    if mc_type == "alpha":
                        for i in range(num_options):
                            tok_id = self.tokenizer.convert_tokens_to_ids(
                                f"{chr(65+i)}"
                            )
                            answer_dict[i] = pd[tok_id].item()
                    elif mc_type == "yn":
                        assert num_options == 2
                        answer_dict[0] = pd[
                            self.tokenizer.convert_tokens_to_ids("yes")
                        ].item()
                        answer_dict[1] = pd[
                            self.tokenizer.convert_tokens_to_ids("no")
                        ].item()

                    sum_pd = sum(answer_dict.values())
                    for i in range(num_options):
                        answer_dict[i] /= sum_pd

                    p_mass = 1 - sum_pd

                    del output
                    torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                logger.error("Failed again after clearing cache. Skipping this prompt.")

        return answer_dict, p_mass
