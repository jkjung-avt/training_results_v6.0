# Copyright (c) 2024-2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import json
from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DatasetProcessor:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def convert_split(self, split: str) -> None:
        input_file = self.data_dir / "data" / f"{split}-00000-of-00001.parquet"
        output_file = self.data_dir / f"{split}.npy"
        metadata_file = self.data_dir / f"{split}_metadata.jsonl"

        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        try:
            logging.info(f"Converting {split} split...")
            df = pd.read_parquet(input_file)

            transformed_data = []
            metadata_list = []
            for _, row in df.iterrows():
                seq_len = len(row["input_ids"])
                transformed_data.append({
                    "input_ids": row["input_ids"],
                    "loss_mask": [int(x != -100) for x in row["labels"]],  # -100 = PyTorch ingore_index
                    "seq_start_id": [0],
                })
                metadata_list.append({
                    "cu_seqlens": [0, seq_len],
                    "max_samples_per_bin": 1
                })
            with open(metadata_file, "w") as mf:
                json.dump(metadata_list, mf)

            np.save(output_file, transformed_data)
            logging.info(f"Successfully converted {split} split")
        except Exception as e:
            logging.error(f"Error converting {split} split: {e}")
            raise

    def download_and_process(self):
        try:
            logging.info("Downloading dataset...")
            snapshot_download(
                "regisss/scrolls_gov_report_preprocessed_mlperf_2",
                revision="21ff1233ee3e87bc780ab719c755170148aba1cb",
                allow_patterns="*.parquet",
                local_dir=self.data_dir,
                max_workers=8,
                repo_type="dataset",
            )
            logging.info("Dataset download completed")

            for split in ["train", "validation"]:
                self.convert_split(split)

            logging.info("Cleaning up intermediate files...")
            shutil.rmtree(self.data_dir / "data", ignore_errors=True)
            logging.info("Cleanup completed")
            logging.info("Dataset processing completed successfully")
        except Exception as e:
            logging.error(f"Dataset processing failed: {e}")
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/data", type=str, help="Path to the dataset location")
    args = parser.parse_args()

    processor = DatasetProcessor(args.data_dir)
    processor.download_and_process()

if __name__ == "__main__":
    main()
