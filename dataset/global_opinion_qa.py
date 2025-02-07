from .dataset import Dataset

import pandas as pd
import logging
from typing import List

logger = logging.getLogger(__name__)


class GlobalOpinionQADataset(Dataset):
    """
    Load the global opinion QA dataset (originally downloaded from GitHub)
    Dataset source: https://huggingface.co/datasets/Anthropic/llm_global_opinions
    """

    def __init__(self, args):
        super().__init__(args)
        self.name = "global_opinion_qa"
        self.source = (
            "hf://datasets/Anthropic/llm_global_opinions/data/global_opinions.csv"
        )

    def load_dataset(self, category, mode="full"):
        """
        Load the global opinion QA dataset from the Hugging Face datasets library, or from disk if the only_eval flag is set to True.
        This method also shuffles the dataset if the shuffle flag is set to True in the args and limits the number of samples if specified.
        """
        # load the dataset for inference
        self.load_dataset_for_inference()

        self.limit_samples()

        if self.args.shuffle:
            self.subsets = self.subsets.sample(random_state=self.args.seed, frac=1)

    def load_dataset_for_inference(self):
        """
        Load the global opinion QA dataset without annotations from the Hugging Face datasets library.
        """

        df = pd.read_csv(self.source)

        # preprocess the dataset
        df = self.preprocess_global_opinion_qa(df)
        df = df.drop(columns=["options", "question"])

        # load as ray dataset
        self.subsets = df

    def preprocess_global_opinion_qa(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the global opinion QA dataset to format the questions and options as prompts and store the number of options for each question.
        """
        prompts = []
        num_options = []

        for i in range(len(df)):
            # parse the opåtions and selections as lists
            df.iloc[i].options = eval(df.iloc[i].options)

            # make sure all questions are strings
            df.iloc[i].question = str(df.iloc[i].question)

            # format questions and options into prompts
            prompt = (
                df.iloc[i].question
                + "\nHere are the options:\n"
                + self.format_options(df.iloc[i].options)
            )
            prompts.append(prompt)
            num_options.append(len(df.iloc[i].options))

        df.loc[:, "prompt_text"] = prompts
        df.loc[:, "num_answer_options"] = num_options

        return df

    def format_options(self, options: List[str]) -> str:
        """
        Format the options as a string, printing one option per line and adding (A), (B), etc. in front of each option.
        Concatenate the instructions at the end.
        """
        return "\n".join([f"({chr(65 + i)}) {options[i]}" for i in range(len(options))])
