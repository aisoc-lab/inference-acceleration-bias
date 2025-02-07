from .dataset import Dataset

import pandas as pd
from omegaconf import ListConfig
import os


class WorldBenchDataset(Dataset):
    """
    Load the worldbench dataset (originally downloaded from GitHub)
    Dataset source: https://github.com/mmoayeri/world-bench/tree/main
    The dataset was manually preprocessed to work without instruction format prompts.
    """

    def __init__(self, args):
        super().__init__(args)
        self.name = "worldbench"
        self.source = (
            "/path/to/new_worldbench.csv"
        )

    def load_dataset(self, category="all", mode="full"):

        assert os.path.exists(
            self.source
        ), f"Data path {self.source} does not exist. Please download the dataset and check the source path."

        self.load_dataset_for_inference(category)

        self.limit_samples()

        if self.args.shuffle:
            self.subsets = self.subsets.sample(random_state=self.args.seed, frac=1)

    def load_dataset_for_inference(self, category: str):
        df = pd.read_csv(self.source)

        # rename column prompt to prompt_text
        df = df.rename(columns={"prompt": "prompt_text"})

        df["example"] = df["example"].apply(lambda x: eval(x))

        # if category is a string, it should be "all" or a valid category
        if isinstance(category, str):
            assert (
                category == "all" or category in df["category"].unique()
            ), f"Category {category} not found in dataset!"

            if category != "all":
                assert (
                    category in df["category"].unique()
                ), f"Category {category} not found in dataset!"
                df = df[df["category"] == category]

        if isinstance(category, ListConfig):
            assert all(
                [cat in df["category"].unique() for cat in category]
            ), f"Category {category} not found in dataset!"
            df = df[df["category"].isin(category)]

        self.subsets = df
