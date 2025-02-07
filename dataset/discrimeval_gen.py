from .dataset import Dataset

import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)


class DiscrimEvalGenDataset(Dataset):
    """
    Load the modified discrimin-eval dataset for longer generations
    """

    def __init__(self, args):
        super().__init__(args)
        self.name = "discrimeval"
        # using the explicit subset of discrimeval
        self.source = "/path/to/discrim_eval_gen.csv"

    def load_dataset(self, category, mode="full"):
        """
        Load the DiscrimEval dataset from the Hugging Face datasets library, or from disk if the only_eval flag is set to True.
        This method also shuffles the dataset if the shuffle flag is set to True in the args and limits the number of samples if specified.
        """
        assert os.path.exists(
            self.source
        ), f"Data path {self.source} does not exist. Please download the dataset and check the source path."

        # load the dataset for inference
        self.load_dataset_for_inference()

        # only use the specified category
        self.subsets = self.subsets[self.subsets["bias_type"] == category]

        self.limit_samples()

        if self.args.shuffle:
            self.subsets = self.subsets.sample(random_state=self.args.seed, frac=1)

    def load_dataset_for_inference(self):
        """
        Load the DiscrimEval dataset without annotations from the Hugging Face datasets library.
        """

        df = pd.read_csv(self.source)
        # rename column prompt to prompt_text
        df = df.rename(columns={"prompt": "prompt_text"})

        # load as ray dataset
        self.subsets = df
