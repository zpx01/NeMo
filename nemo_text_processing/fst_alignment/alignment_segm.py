# from nemo_text_processing.text_normalization.utils_audio_based import _get_alignment, get_semiotic_spans
import json
import re
import string

import pynini
from alignment import (
    _get_original_index,
    create_symbol_table,
    get_string_alignment,
    get_word_segments,
    indexed_map_to_output,
)
from nemo_text_processing.text_normalization.normalize import Normalizer
from pynini.lib.rewrite import top_rewrite

from nemo.collections.common.tokenizers.moses_tokenizers import MosesProcessor

# cache_dir = "/home/ebakhturina/NeMo/nemo_text_processing/text_normalization/cache_dir"
from nemo.utils import logging


def remove_punctuation(text, remove_spaces=True, do_lower=True, exclude=None):
    all_punct_marks = string.punctuation

    if exclude is not None:
        for p in exclude:
            all_punct_marks = all_punct_marks.replace(p, "")

        # a weird bug where commas is getting deleted when dash is present in the list of punct marks
        all_punct_marks = all_punct_marks.replace("-", "")
    text = re.sub("[" + all_punct_marks + "]", " ", text)

    if exclude and "-" not in exclude:
        text = text.replace("-", " ")

    text = re.sub(r" +", " ", text)
    if remove_spaces:
        text = text.replace(" ", "").replace("\u00A0", "").strip()

    if do_lower:
        text = text.lower()
    return text.strip()


text = "of getting it done this year or at worst case very early into calendar year '19. So things are progressing well there. We also filed for the key state and federal approvals in 2017, and we'll go over that with a chart in a second. We also expect to file rate cases by May for both NYSEG and RG&E for electric and gas by May. The contracts now are expected to be approved by the second quarter of '19, and this just puts it in context with the time frame we expected all along. We received our FERC approval That was filed in September of 2017, and we should have that by mid-2019. And then local and municipal construction approvals will be timed as needed throughout the project. Good afternoon and 5.3"
norm = "of getting it done this year or at worst case very early into calendar year 'nineteen. So things are progressing well there. We also filed for the key state and federal approvals in twenty seventeen, and we'll go over that with a chart in a second. We also expect to file rate cases by May for both NYSEG and RG&E for electric and gas by May. The contracts now are expected to be approved by the second quarter of 'nineteen, and this just puts it in context with the time frame we expected all along. We received our FERC approval That was filed in September of twenty seventeen, and we should have that by mid- twenty nineteen. And then local and municipal construction approvals will be timed as needed throughout the project. Good afternoon and five point three"


segmented = [
    "of getting it done this year or at worst case very early into calendar year 'nineteen.",
    "So things are progressing well there. We also filed for the key state and federal approvals",
    "in twenty seventeen, and we'll go over that with a chart in a second. We also expect to file rate cases by May",
    "for both NYSEG and RG and E for electric and gas by May. The contracts now are expected to be approved by the second",
    "quarter of 'nineteen, and this just puts it in context with the time frame we expected all along.",
    "We received our FERC approval That was filed in September of twenty seventeen, and we should have that by",
    "mid twenty nineteen. And then local and municipal construction approvals will be timed as needed throughout the project. Good afternoon and five",
    "point three",
]

data_dir = "/media/ebakhturina/DATA/mlops_data/ds_align_data"
data_dir = "/Users/ebakhturina/Downloads/ds_align_data"
raw_manifest = f"{data_dir}/raw.json"
segmented_manifest = f"{data_dir}/segmented.json"

with open(raw_manifest, "r") as f_in:
    for line in f_in:
        line = json.loads(line)
        text = line["text"]
        text = remove_punctuation(text, do_lower=True, remove_spaces=False, exclude="'")

segmented = []
with open(segmented_manifest, "r") as f_in:
    for line in f_in:
        line = json.loads(line)
        segmented.append(line["text"])

# segmented_idx = []
#
# start_idx = 0
# for segment in segmented:
#     start_idx = norm.index(segment, start_idx)
#     segmented_idx.append([start_idx, start_idx + len(segment)])
#     start_idx += len(segment)

# for idx in segmented_idx:
#     print(norm[idx[0]: idx[1]])
# import pdb; pdb.set_trace()


# logging.setLevel("DEBUG")
cache_dir = "cache_dir"
lang = "en"
normalizer = Normalizer(input_case="cased", cache_dir=cache_dir, overwrite_cache=False, lang=lang)

moses_processor = MosesProcessor(lang_id=lang)
# fst = normalizer.tagger.fst
fst = pynini.compose(normalizer.tagger.fst, normalizer.verbalizer.fst)
# fst = pynini.compose(fst, normalizer.post_processor.fst)
table = create_symbol_table()
# text = "Hi, the total is $13.5. Please pay using card, in '19"
alignment, output_text = get_string_alignment(fst=fst, input_text=text, symbol_table=table)
# output_text = moses_processor.moses_detokenizer.detokenize([output_text], unescape=False)

indices = get_word_segments(text)


def punct_post_process(text):
    text = top_rewrite(text, normalizer.post_processor.fst)
    text = moses_processor.moses_detokenizer.detokenize([text], unescape=False)
    return text

    # text = (text.replace('tokens { name: "', "")
    # 		.replace('" }', "")
    # 		.replace('preserve_order: true }', '')
    # 		.replace('preserve_order: false }', '')
    # 		.replace(' } ', '')
    # 		.replace("  ", " ")
    # 		.replace(" .", ".")
    # 		.replace(" ,", ","))
    # return text


# print(output_text)
# print()
# for x in indices:
#     start, end = indexed_map_to_output(start=x[0], end=x[1], alignment=alignment)
#     # print(f"[{x[0]}:{x[1]}] -- [{start}:{end}]")
#     print(f"|{text[x[0]:x[1]]}| -- |{output_text[start:end]}|")
#
# import pdb; pdb.set_trace()
segment_id = 0

restored = []
text_list = text.split()

cur_segment = ""
last_segment_word = segmented[segment_id]

start_idx = 0
end_idx = 1
# import pdb
#
# pdb.set_trace()
while end_idx < len(text):
    out_start, out_end = indexed_map_to_output(start=start_idx, end=end_idx, alignment=alignment)
    aligned_output = output_text[out_start:out_end]

    aligned_output = punct_post_process(aligned_output)
    if aligned_output == segmented[segment_id]:
        restored.append(aligned_output)
        print("=" * 40)
        print(text[start_idx:end_idx])
        print(segmented[segment_id])
        print(aligned_output)
        print("=" * 40)
        # import pdb
        #
        # pdb.set_trace()
        print()
        segment_id += 1
        start_idx = end_idx
        while start_idx < len(text) and text[start_idx] == " ":
            start_idx += 1
        # print()
    end_idx += 1

[print(x) for x in restored]
import pdb

pdb.set_trace()
print()
# # input_text = args.text


# norm_raw_diffs = _get_alignment(norm, raw)
#
# for k, v in norm_raw_diffs.items():
#     a = norm.split()
#     b = raw.split()
#     print(f"{k}: {a[k]} -- {b[v[0]:v[1]]} -- {v[2]}")


# semiotic_spans, pred_texts, norm_spans, raw_text_masked_list, raw_text_mask_idx = adjust_boundaries(
# 	norm_raw_diffs, norm_pred_diffs, raw, norm, pred_text
# )
#
# args = parse_args()
# fst = Far(args.fst, mode='r')['tokenize_and_classify']
# input_text = "He paid 25 dollars in 1980"
