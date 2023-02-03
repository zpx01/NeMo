import argparse
import json
import os
from glob import glob
from typing import List, Tuple

SCORE_PAD = 0

parser = argparse.ArgumentParser(
    description="Convert text processed with ctc segmentation prepare_data.py to NFA format"
)
parser.add_argument(
    "--nfa_manifest", type=str, required=True, help="Path to output .json manifest with NFA alignment metadata"
)
parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
    help="Path to directory to score NFA alignments in CTC segmentation format",
)


def write_output(
    out_path: str,
    path_wav: str,
    segments: List[Tuple[float]],
    text: str,
    text_no_preprocessing: str,
    text_normalized: str,
):
    """
	Write the segmentation output to a file

	out_path: Path to output file
	path_wav: Path to the original audio file
	segments: Segments include start, end and alignment score
	text: Text used for alignment
	text_no_preprocessing: Reference txt without any pre-processing
	text_normalized: Reference text normalized
	"""
    # Uses char-wise alignments to get utterance-wise alignments and writes them into the given file
    with open(str(out_path), "w") as outfile:
        outfile.write(str(path_wav) + "\n")

        for i, segment in enumerate(segments):
            if isinstance(segment, list):
                for j, x in enumerate(segment):
                    start, duration = x
                    score = -100
                    outfile.write(
                        f"{start} {start + duration} {score} | {text[i][j]} | {text_no_preprocessing[i][j]} | {text_normalized[i][j]}\n"
                    )
            else:
                start, end, score = segment
                outfile.write(
                    f"{start} {end} {score} | {text[i]} | {text_no_preprocessing[i]} | {text_normalized[i]}\n"
                )


def read_text_files(transcript_file):
    if not os.path.exists(transcript_file):
        raise ValueError(f"{transcript_file} not found.")

    with open(transcript_file, "r") as f:
        text = f.readlines()
        text = [t.strip() for t in text if t.strip()]

    # add corresponding original text without pre-processing
    transcript_file_no_preprocessing = transcript_file.replace(".txt", "_with_punct.txt")
    if not os.path.exists(transcript_file_no_preprocessing):
        raise ValueError(f"{transcript_file_no_preprocessing} not found.")

    with open(transcript_file_no_preprocessing, "r") as f:
        text_no_preprocessing = f.readlines()
        text_no_preprocessing = [t.strip() for t in text_no_preprocessing if t.strip()]

    # add corresponding normalized original text
    transcript_file_normalized = transcript_file.replace(".txt", "_with_punct_normalized.txt")
    if not os.path.exists(transcript_file_normalized):
        raise ValueError(f"{transcript_file_normalized} not found.")

    with open(transcript_file_normalized, "r") as f:
        text_normalized = f.readlines()
        text_normalized = [t.strip() for t in text_normalized if t.strip()]

    if len(text_no_preprocessing) != len(text):
        raise ValueError(f"{transcript_file} and {transcript_file_no_preprocessing} do not match")

    if len(text_normalized) != len(text):
        raise ValueError(f"{transcript_file} and {transcript_file_normalized} do not match")
    return text, text_normalized, text_no_preprocessing


def extract_time_stamps(additional_segment_level_ctm_filepath, audio_filepath, output_dir):
    if not os.path.exists(additional_segment_level_ctm_filepath):
        raise ValueError(f"{additional_segment_level_ctm_filepath} not found")

    filename = os.path.splitext(audio_filepath)[0]
    filename = f"{filename}.txt"

    output_file = f"{output_dir}/{os.path.splitext(os.path.basename(audio_filepath))[0]}_segments.txt"

    text, text_normalized, text_no_preprocessing = read_text_files(filename)

    with open(additional_segment_level_ctm_filepath, "r") as f:
        nfa_segments = f.readlines()
        if len(nfa_segments) != len(text):
            raise ValueError("Number of NFA segments do not match the original number of text segments")

    with open(output_file, "w") as f_out:
        f_out.write(f"{audio_filepath}\n")
        for i, nfa_segment in enumerate(nfa_segments):
            nfa_segment = nfa_segment.split()
            start, duration = float(nfa_segment[2]), float(nfa_segment[3])
            f_out.write(
                f"{start} {start + duration} {SCORE_PAD} | {text[i]} | {text_no_preprocessing[i]} | {text_normalized[i]}\n"
            )


if __name__ == "__main__":
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.nfa_manifest):
        raise ValueError(f"{args.nfa_manifest} not found")

    with open(args.nfa_manifest, "r") as f:
        for line in f:
            line = json.loads(line)
            extract_time_stamps(line["additional_segment_level_ctm_filepath"], line["audio_filepath"], args.output_dir)
