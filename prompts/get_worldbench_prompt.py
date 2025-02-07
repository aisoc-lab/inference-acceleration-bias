from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer


MODEL_PATHS = {
    "llama2chat": "meta-llama/Llama-2-7b-chat-hf",
    "llama3chat": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "mistralchat": "mistralai/Mistral-7B-Instruct-v0.3",
}


class GetWorldBenchPrompt:
    """
    Constructs a LLM prompt from an input data item according to a template.
    """

    def __init__(
        self,
        use_chat_template: bool,
        prompt_template: str,
        system_message: str,
        model_name: str,
        answer_prefix: str,
    ):
        """
        Initializes the prompt generator with the prompt template.

        Args:
            use_chat_template: Whether to use the chat template for the model.
            prompt_template: The template for the prompt. Must contain the string "$model_input".
            system_message: The system message to prepend to the prompt.
            model_name: The name of the model to use [llama2chat, llama3chat, mistralchat].
            answer_prefix: The prefix to append to the model's answer to the prompt.
        """
        self.prompt_template = prompt_template
        self.system_message = system_message
        self.use_chat_template = use_chat_template
        self.answer_prefix = answer_prefix

        model_link = MODEL_PATHS[model_name]
        self.tokenizer = AutoTokenizer.from_pretrained(model_link)

    def compose_prompt(self, data_input, example, metric):
        sys_message = self.system_message.replace("$metric", metric)

        prompt_text = self.prompt_template.replace(f"$model_input", data_input)

        answer_prefix = self.answer_prefix

        if not self.use_chat_template:
            prompt = ""
            if sys_message:
                prompt += sys_message + " "
            if example:
                prompt += f"{example[0]} {example[1]} \n"
            prompt += prompt_text
            if answer_prefix:
                prompt += "\n\n" + answer_prefix

        else:
            messages = []
            if sys_message:
                messages.append(
                    {"role": "user", "content": sys_message}
                )  # NOTE: here, the role is "user" instead of "system"
                messages.append(
                    {"role": "assistant", "content": f"Sounds good, will do."}
                )

            if example:
                messages.append({"role": "user", "content": str(example[0])})
                messages.append({"role": "assistant", "content": str(example[1])})

            messages.append({"role": "user", "content": prompt_text})

            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            if answer_prefix:
                answer_prefix = answer_prefix.strip()
                prompt += " " + answer_prefix

        return prompt

    def __call__(self, input_prompt: str, example: Tuple, metric: str) -> str:
        """
        Augments the input data item with the generated prompt and returns the augmented data item.

        Args:
            input_prompt: The input data item to be augmented with the generated prompt.
            example: The example to be included in the prompt.
            metric: The metric to be included in the prompt.


        Returns:
            The augmented data item with the generated prompt added.
        """
        out = self.compose_prompt(input_prompt, example, metric)
        return out
