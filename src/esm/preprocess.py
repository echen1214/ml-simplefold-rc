from pathlib import Path
from src.esm.model.RCfold import AlignBio_DataModule
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data for AlignBio_DataModule")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the data directory")
    parser.add_argument("--esm_cache_dir", type=str, required=True, help="Path to the ESM cache directory")
    parser.add_argument("--csv", type=str, required=True, help="Path to the CSV file")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (default: 8)")

    args = parser.parse_args()

    datamodule = AlignBio_DataModule(
        Path(args.data_dir),
        Path(args.csv),
        Path(args.esm_cache_dir),
        preprocess=True,
        batch_size=args.batch_size
    )