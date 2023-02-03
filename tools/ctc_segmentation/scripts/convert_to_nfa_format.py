import argparse
import json
import os
from glob import glob

parser = argparse.ArgumentParser(
    description="Convert text processed with ctc segmentation prepare_data.py to NFA format"
)
parser.add_argument(
    "--processed_data",
    type=str,
    required=True,
    help="Path to a directory with processed text files (text is split rough by senteces, normalized, symbols not present in the ASR vocab are removed)",
)
parser.add_argument("--out_manifest", type=str, required=True, help="Path to output .json manifest")

if __name__ == "__main__":
    args = parser.parse_args()

    if not os.path.exists(args.processed_data):
        raise ValueError(f"{args.processed_dat} not found")

    files = set(
        [
            f.replace("_with_punct_normalized", "").replace("_with_punct", "")
            for f in glob(f"{args.processed_data}/*.txt")
        ]
    )

    with open(args.out_manifest, "w") as f_out:
        for file in files:
            with open(file, "r") as f_text:
                text = f_text.read().replace("\n", " | ").replace("  ", " ")
                line = {"audio_filepath": file.replace(".txt", ".wav"), "text": text.strip()}
                f_out.write(json.dumps(line, ensure_ascii=False) + '\n')
