# from nemo_text_processing.text_normalization.utils_audio_based import _get_alignment, get_semiotic_spans
import json
import os
import pickle
import re
from time import perf_counter

import pynini
from alignment import (
    EPS,
    _get_original_index,
    create_symbol_table,
    get_string_alignment,
    get_word_segments,
    indexed_map_to_output,
    remove,
)
from joblib import Parallel, delayed
from nemo_text_processing.text_normalization.normalize import Normalizer
from pynini.lib.rewrite import top_rewrite
from tqdm import tqdm

from nemo.collections.common.tokenizers.moses_tokenizers import MosesProcessor

# cache_dir = "/home/ebakhturina/NeMo/nemo_text_processing/text_normalization/cache_dir"
from nemo.utils import logging

NA = "n/a"


def clean(text):
    text = text.replace(":", " ").replace("-", " ").replace(" .", ".").replace(" ,", ",").replace(" ?", "?")

    text = text.replace("  ", " ")
    return text


def build_output_raw_map(alignment, output_text, text):
    # get TN alignment
    indices = get_word_segments(text)
    output_raw_map = []

    for i, x in enumerate(indices):
        try:
            start, end = indexed_map_to_output(start=x[0], end=x[1], alignment=alignment)
        except:
            print(f"{key} -- error")
            return None

    output_raw_map.append([output_text[start:end], text[x[0] : x[1]]])
    return output_raw_map


def process_with_normalizer(item, key, normalizer, fst, offset=5, output_raw_map=None, use_cache=True):
    restored = []
    text = item["raw_text"]

    if "segmented" not in item:
        return (key, restored)

    segmented = item["segmented"]

    if output_raw_map is None:
        if os.path.exists(f"alignment_{key}.p") and use_cache:
            alignment = pickle.load(open(f"alignment_{key}.p", "rb"))
            output_text = pickle.load(open(f"output_text_{key}.p", "rb"))
        else:
            alignment, output_text = get_string_alignment(fst=fst, input_text=text, symbol_table=table)
            pickle.dump(alignment, open(f"alignment_{key}.p", "wb"))
            pickle.dump(output_text, open(f"output_text_{key}.p", "wb"))
            print("tn output saved")

        restored = []
        output_raw_map = build_output_raw_map(alignment, output_text, text)
        if output_raw_map is None:
            return (key, restored)

    last_found_start_idx = 0
    for segment_id in range(0, len(segmented)):
        restored_raw = NA
        segment = segmented[segment_id]
        segment_list = segment.split()

        end_found = False
        if len(segment_list) == 0:
            restored.append(restored_raw)
            continue
        first_word = segment_list[0]
        for id in [
            i for i, x in enumerate(output_raw_map[last_found_start_idx:]) if first_word.lower() in x[0].lower()
        ]:
            if end_found:
                break
            end_idx = id + (len(segment_list) - offset)
            while not end_found and end_idx <= len(output_raw_map):
                restored_norm = " ".join([x[0] for x in output_raw_map[last_found_start_idx:][id:end_idx]])
                restored_raw = " ".join([x[1] for x in output_raw_map[last_found_start_idx:][id:end_idx]])

                processed_raw = clean(normalizer.normalize(restored_raw).lower())
                processed_segment = clean(segment.lower())
                processed_restored = clean(restored_norm.lower())
                if processed_restored == processed_segment or processed_raw == processed_segment:
                    end_found = True
                    last_found_start_idx = end_idx
                elif processed_segment.startswith(processed_restored) or processed_segment.startswith(processed_raw):
                    end_idx += 1
                else:
                    restored_raw = NA
                    break

        restored.append(restored_raw)
    return (key, restored)


def build_output_raw_map(alignment, output_text, text):
    indices = get_word_segments(text)
    output_raw_map = []

    for i, x in enumerate(indices):
        try:
            start, end = indexed_map_to_output(start=x[0], end=x[1], alignment=alignment)
        except:
            print(f"{key} -- error")
            return None

        output_raw_map.append([output_text[start:end], text[x[0] : x[1]]])
    return output_raw_map


def get_raw_text_from_alignment(alignment, alignment_start_idx=0, alignment_end_idx=None):
    if alignment_end_idx is None:
        alignment_end_idx = len(alignment)

    return "".join(list(map(remove, [x[0] for x in alignment[alignment_start_idx : alignment_end_idx + 1]])))


def process(item, key, normalizer, fst, use_cache=True, verbose=False):
    restored = []
    text = item["raw_text"]

    if "segmented" not in item:
        return (key, restored)

    segmented = item["segmented"]
    cached_alignment = f"alignment_{key}.p"
    if os.path.exists(cached_alignment) and use_cache:
        alignment = pickle.load(open(f"alignment_{key}.p", "rb"))
    else:
        alignment, _ = get_string_alignment(fst=fst, input_text=text, symbol_table=table)
        pickle.dump(alignment, open(f"alignment_{key}.p", "wb"))
        print(f"alignment output saved to {cached_alignment}")

    segmented_result = []
    segmented_indices = []
    for i, x in enumerate(alignment):
        value = remove(x[1])
        if value != "":
            segmented_result.append(value)
            segmented_indices.append(i)

    segmented_result = "".join(segmented_result)
    failed = []
    alignment_end_idx = None
    for id, segment in enumerate(segmented):
        if segment.lower() in segmented_result.lower():
            segment_start_idx = segmented_result.lower().index(segment.lower())
            alignment_start_idx = segmented_indices[segment_start_idx]
            alignment_end_idx = segmented_indices[segment_start_idx + len(segment) - 1]

            if verbose:
                raw_text = "".join(
                    list(map(remove, [x[0] for x in alignment[alignment_start_idx : alignment_end_idx + 1]]))
                )
                print("=" * 40)
                print("FOUND:")
                print(f"RAW : {raw_text}")
                print(f"SEGM: {segment}")
                print("=" * 40)

            if len(failed) > 0 and len(failed[-1]) == 3:
                failed[-1].append(alignment_start_idx)
                idx = len(failed) - 2
                while idx >= 0 and len(failed[idx]) == 3:
                    failed[idx].append(alignment_start_idx)
                    idx -= 1
            # alignment_end_idx += 1
        elif alignment_end_idx is not None:
            failed.append([id, segment, alignment_end_idx])
            if id == len(segmented) - 1:
                failed[-1].append(len(alignment))
                idx = len(failed) - 2
                while idx >= 0 and len(failed[idx]) == 3:
                    failed[idx].append(alignment_start_idx)
                    idx -= 1

    failed_restored = []
    for i in range(len(failed)):
        alignment_start_idx, alignment_end_idx = failed[i][2], failed[i][3]
        raw_text_ = get_raw_text_from_alignment(
            alignment, alignment_start_idx=alignment_start_idx, alignment_end_idx=alignment_end_idx
        )
        alignment_current = alignment[alignment_start_idx : alignment_end_idx + 1]
        output_norm_current = "".join(map(remove, [x[1] for x in alignment_current]))
        item = {"raw_text": raw_text_, "segmented": [failed[i][1]], "misc": ""}

        output_raw_map = build_output_raw_map(alignment_current, output_norm_current, raw_text_)

        if output_raw_map is None:
            continue

        failed_restored.append(
            process_with_normalizer(item, "debug", normalizer, fst, output_raw_map=output_raw_map, use_cache=False)
        )

        if failed_restored[0][-1][0] == NA:
            print("=" * 40)
            print(f"RAW : {raw_text_}")
            print(f"SEGM: {failed[i][1]}")
            print("=" * 40)

    import pdb

    pdb.set_trace()
    print()


if __name__ == "__main__":
    raw_text = ""
    norm_text = ""
    data_dir = "/media/ebakhturina/DATA/mlops_data/pc_retained"
    raw_manifest = f"{data_dir}/raw.json"
    segmented_manifest = f"{data_dir}/segmented.json"

    # for every audio file store "raw" and "segmented" samples
    # data["Am4BKyvYBgY"].keys()
    # >> dict_keys(['raw_text', 'segmented'])

    data = {}
    # read raw manifest
    with open(raw_manifest, "r") as f_in:
        for line in f_in:
            line = json.loads(line)
            text = re.sub(r" +", " ", line["text"])
            audio = line["audio_filepath"].split("/")[-1].replace(".wav", "")
            data[audio] = {"raw_text": text}

    segmented = {}
    misc_data = {}
    # read manifest after segmentation
    with open(segmented_manifest, "r") as f_in:
        for line in f_in:
            line = json.loads(line)
            text = re.sub(r" +", " ", line["text"])
            audio = "_".join(line["audio_filepath"].split("/")[-1].split("_")[:-1])

            if audio not in segmented:
                segmented[audio] = []
                misc_data[audio] = []
            segmented[audio].append(line["text"])
            misc_data[audio].append(line)

    for audio in segmented:
        segmented_lines = segmented[audio]
        misc_line_data = misc_data[audio]

        if audio not in data:
            print(f"{audio} from {segmented_manifest} is missing in the {raw_manifest}")
        else:
            data[audio]["segmented"], data[audio]["misc"] = [], []
            for segm, misc in zip(segmented_lines, misc_line_data):
                if len(segm) > 0:
                    data[audio]["segmented"].append(segm)
                    data[audio]["misc"].append(misc)

    # remove data where there are no corresponding segmented samples
    audio_data_to_del = [audio for audio in data if "segmented" not in data[audio].keys()]
    print(f"No corresponding segments found for {audio_data_to_del}, removing")
    for key in audio_data_to_del:
        del data[key]

    # normalize raw manifest for alignment
    cache_dir = "cache_dir"
    lang = "en"
    normalizer = Normalizer(input_case="cased", cache_dir=cache_dir, overwrite_cache=False, lang=lang)

    # moses_processor = MosesProcessor(lang_id=lang)
    # fst = normalizer.tagger.fst
    fst = pynini.compose(normalizer.tagger.fst, normalizer.verbalizer.fst)
    fst = pynini.compose(fst, normalizer.post_processor.fst)
    table = create_symbol_table()

    verbose = True
    all_results = {}
    with open("log.txt", "w") as f:
        for key, item in tqdm(data.items()):
            if key == "AJ4D8gciKb0":
                try:
                    all_results[key] = process(item, key, normalizer, fst)[1]
                    not_found = [(i, r) for i, r in enumerate(all_results[key]) if r == NA]
                    num_not_found = len(not_found)
                    f.write(
                        f'{key} -- number not found: {num_not_found} out of {len(all_results[key])} ({round(num_not_found/len(all_results[key])*100, 1)}%)'
                    )
                    f.write("\n")

                    if verbose:
                        print("=" * 40)
                        print(
                            f'{key} -- number not found: {num_not_found} out of {len(all_results[key])} ({round(num_not_found / len(all_results[key]) * 100, 1)}%)'
                        )

                        if num_not_found > 0:
                            for idx, _ in not_found:
                                print(data[key]["segmented"][idx])
                        print("=" * 40)
                except Exception as e:
                    print(f"{key} -- FAILED -- {e}")
                    raise e
                    import pdb

                    pdb.set_trace()
                    f.write(f"{key} -- FAILED -- {e}")
                    f.write("\n")
                    raise e

    import pdb

    pdb.set_trace()
    print()

    # results = Parallel(n_jobs=16)(delayed(process_file)(item, key) for key, item in tqdm(data.items()))

    print("DONE with parallel")

    import pdb

    pdb.set_trace()
    print()
