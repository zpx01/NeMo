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
    EPS
)
from nemo_text_processing.text_normalization.normalize import Normalizer
from pynini.lib.rewrite import top_rewrite

from nemo.collections.common.tokenizers.moses_tokenizers import MosesProcessor
from joblib import Parallel, delayed

# cache_dir = "/home/ebakhturina/NeMo/nemo_text_processing/text_normalization/cache_dir"
from nemo.utils import logging


raw_text = ""
norm_text = ""
data_dir = "/media/ebakhturina/DATA/mlops_data/pc_retained"
raw_manifest = f"{data_dir}/raw_sample.json"
segmented_manifest = f"{data_dir}/segmented_sample.json"

data = {}
with open(raw_manifest, "r") as f_in:
    for line in f_in:
        line = json.loads(line)
        text = line["text"]
        audio = line["audio_filepath"].split("/")[-1].replace(".wav", "")
        data[audio] = {"raw_text": text}

segmented = {}
with open(segmented_manifest, "r") as f_in:
    for line in f_in:
        line = json.loads(line)
        audio = line["audio_filepath"].split("/")[-1].split("_")[0]
        if audio not in segmented:
            segmented[audio] = []
        segmented[audio].append(line["text"])

for audio, segmented_lines in segmented.items():
    if audio not in data:
        import pdb; pdb.set_trace()
    else:
        data[audio]["segmented"] = segmented_lines

text = data["ASbInYbN1Sc"]["raw_text"]
segmented = data["ASbInYbN1Sc"]["segmented"]

import pickle

"""
table = create_symbol_table()
cache_dir = "cache_dir"
lang = "en"
normalizer = Normalizer(input_case="cased", cache_dir=cache_dir, overwrite_cache=False, lang=lang)

moses_processor = MosesProcessor(lang_id=lang)
# fst = normalizer.tagger.fst
fst = pynini.compose(normalizer.tagger.fst, normalizer.verbalizer.fst)
# fst = pynini.compose(fst, normalizer.post_processor.fst)
table = create_symbol_table()
alignment, output_text = get_string_alignment(fst=fst, input_text=text, symbol_table=table)

pickle.dump(alignment, open("alignment.p", "wb" ) )
pickle.dump(output_text, open("output_text.p", "wb" ) )
"""

alignment = pickle.load(open("alignment.p", "rb" ))
output_text = pickle.load( open("output_text.p", "rb"))


indices = get_word_segments(text)

output_raw_map = []
for i, x in enumerate(indices):
    start, end = indexed_map_to_output(start=x[0], end=x[1], alignment=alignment)
    output_raw_map.append([output_text[start:end], text[x[0]:x[1]]])
    # print(f"inp indices: [{x[0]}:{x[1]}] out indices: [{start}:{end}]")
    # print(f"in: |{text[x[0]:x[1]]}| out: |{output_text[start:end]}|")


import re

def find_segment(segment_id):
    restored_raw = None
    idx = 0
    offset = 5

    norm_text = " ".join([x[0] for x in output_raw_map])
    # norm_text_idx = " ".join([i for i in range(len(output_raw_map))])
    raw_text = " ".join([x[1] for x in output_raw_map])
    cur_segment = segmented[segment_id].split()
    end_match_found = False

    pattern = cur_segment[0]
    max_start_match_len = min(4, len(cur_segment))
    for i in range(1, max_start_match_len):
        pattern += f"[^A-Za-z]+{cur_segment[i]}"

    pattern = re.compile(pattern)

    for i, m in enumerate(pattern.finditer(norm_text)):
        if end_match_found:
            break

        match_idx = m.start()
        stop_end_search = False
        end_idx = len(cur_segment) - offset
        norm_text_list = norm_text[match_idx:].split()
        # import pdb; pdb.set_trace()
        while not end_match_found and end_idx <= len(norm_text_list) and not stop_end_search:
            restored = " ".join(norm_text_list[:end_idx])
                # " ".join([x[0] for x in output_raw_map[idx: end_idx]])
            # print("segm: ", segmented[segment_id])
            # print("rest: ", restored)
            # import pdb;
            # pdb.set_trace()
            if restored == segmented[segment_id]:
                stop_end_search = True
                end_match_found = True
                restored_raw = " ".join(raw_text[match_idx:].split()[:end_idx])
            else:
                end_idx += 1
                if end_idx > (len(cur_segment) + offset):
                    stop_end_search = True

    # print(restored)
    # import pdb; pdb.set_trace()
    # print()

    # for word in segmented[segment_id].split():
    #     while idx < len(output_raw_map) and not found:
    #         if output_raw_map[idx][0].startswith(word) and not found_start:
    #             end_idx = idx + len(segmented[segment_id].split()) - offset
    #             restored = " ".join([x[0] for x in output_raw_map[idx: end_idx]])
    #             max_len = min(offset * 2, len(segmented[segment_id]))
    #             # check that the start matches
    #             if restored[:max_len] == segmented[segment_id][:max_len]:
    #                 found_start = True
    #
    #                 found_end = False
    #                 while end_idx < len(output_raw_map) and (end_idx - idx) < len(segmented[segment_id]) + offset and not found_end:
    #                     restored = " ".join([x[0] for x in output_raw_map[idx: end_idx]])
    #                     print(segmented[segment_id])
    #                     print(restored)
    #                     import pdb; pdb.set_trace()
    #                     if restored == segmented[segment_id]:
    #                         found_end = True
    #                         found = True
    #                     else:
    #                         end_idx += 1
    #             else:
    #
    #         else:
    #             idx += 1

    return (segment_id, restored_raw)


results = Parallel(n_jobs=16)(delayed(find_segment)(segment_id) for segment_id in range(0, len(segmented)))
# results = find_segment(1)

for segment, raw in zip(segmented, results):
    print("segm:", segment)
    print("raw :", raw[1])
    print("+" * 40)
# print(results)
import pdb; pdb.set_trace()
print()