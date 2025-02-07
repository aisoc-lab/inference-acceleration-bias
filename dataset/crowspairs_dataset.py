from .dataset import Dataset
import os
import pandas as pd


class CrowsPairsDataset(Dataset):
    def __init__(self, args) -> None:

        super().__init__(args)
        self.name = "crows_pairs"
        self.source = self.args.source

    def load_dataset(self):
        """
        Load the Crows-Pairs dataset from disk.
        This method also shuffles the dataset if the shuffle flag is set to True in the args.
        """
        assert os.path.exists(
            self.source
        ), f"Data path {self.source} does not exist. Please clone the CrowSPairs repository and check the source path."

        dataset = pd.read_csv(self.source)
        dataset = dataset.drop(
            columns=["anon_annotators", "annotations", "anon_writer"]
        )
        self.subsets = dataset
        # limit the number of samples if we want to
        self.limit_samples()
        # shuffle the dataset
        if self.args.shuffle:
            self.subsets = self.subsets.sample(random_state=self.args.seed, frac=1)
