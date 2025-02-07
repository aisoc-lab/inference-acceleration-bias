from .dataset import Dataset

import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DiscrimEvalDataset(Dataset):
    """
    Load the DiscrimEval dataset (originally downloaded from GitHub)
    Dataset source: https://huggingface.co/datasets/Anthropic/discrim-eval
    """

    def __init__(self, args):
        super().__init__(args)
        self.name = "discrimeval"
        # using the explicit subset
        self.source = "hf://datasets/Anthropic/discrim-eval/explicit.jsonl"

    def load_dataset(self, category, mode="full"):
        """
        Load the DiscrimEval dataset from the Hugging Face datasets library, or from disk if the only_eval flag is set to True.
        This method also shuffles the dataset if the shuffle flag is set to True in the args and limits the number of samples if specified.
        """
        # load the dataset for inference
        self.load_dataset_for_inference()

        self.limit_samples()

        if self.args.shuffle:
            self.subsets = self.subsets.sample(random_state=self.args.seed, frac=1)

    def load_dataset_for_inference(self):
        """
        Load the DiscrimEval dataset without annotations from the Hugging Face datasets library.
        """
        df = pd.read_json(self.source, lines=True)
        # format the prompts and add them to a new column "prompt_text"
        df = self.preprocess_discrim_eval(df)
        df = df.drop(columns=["fill_type", "filled_template"])
        self.subsets = df

    def preprocess_discrim_eval(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the DiscrimEval dataset
        """
        prompts = []

        for i in range(len(df)):
            # format the prompt
            prompt = self.format_prompt(df.loc[i, "filled_template"])
            prompts.append(prompt)

        df.loc[:, "prompt_text"] = prompts
        df.loc[:, "num_answer_options"] = 2

        return df

    def format_prompt(self, item: str) -> str:
        """
        Format the prompt by concatenating the decision prompt at the end.
        """
        prompt_end = """Please answer the above question with either "yes" or "no"."""

        return item + "\n\n" + prompt_end.strip()
