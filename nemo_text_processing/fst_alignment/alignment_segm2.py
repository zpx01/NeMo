# from nemo_text_processing.text_normalization.utils_audio_based import _get_alignment, get_semiotic_spans
import json
import re
import os
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
from tqdm import tqdm
from nemo.collections.common.tokenizers.moses_tokenizers import MosesProcessor
from joblib import Parallel, delayed
import pickle
from time import perf_counter

# cache_dir = "/home/ebakhturina/NeMo/nemo_text_processing/text_normalization/cache_dir"
from nemo.utils import logging

NA = "n/a"


def process(text):
    text = (text.replace(":", " ")
            .replace("-", " ")
            .replace(" .", ".")
            .replace(" ,", ",")
            .replace(" ?", "?"))

    text = text.replace("  ", " ")
    return text

def build_output_raw_map(alignment, output_text, text):
    # get TN alignment
    start_time = perf_counter()
    indices = get_word_segments(text)
    output_raw_map = []

    for i, x in enumerate(indices):
        try:
            start, end = indexed_map_to_output(start=x[0], end=x[1], alignment=alignment)
        except:
            print(f"{key} -- error")
            return None

    output_raw_map.append([output_text[start:end], text[x[0]:x[1]]])
    print(f'FST alignment: {round((perf_counter() - start_time) / 60, 2)} min.')
    return output_raw_map

def process_file(item, key, normalizer, fst, offset=5, output_raw_map=None, use_cache=True):
    # print(f"processing {key}")
    #
    # key = "debug"
    # item = {'raw_text': "On 12/12/2015 and he gave me $5 then and",
    #         'segmented': ["he gave me five dollars then", "and"],
    #         'misc': ""}
    # # import pdb; pdb.set_trace()
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
            pickle.dump(output_text, open(f"output_text_{key}.p", "wb" ))
            print("tn output saved")

        restored = []
        output_raw_map = build_output_raw_map(alignment, output_text, text)
        if output_raw_map is None:
            return (key, restored)

    start_time = perf_counter()
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
        for id in [i for i, x in enumerate(output_raw_map[last_found_start_idx:]) if first_word.lower() in x[0].lower()]:
            if end_found:
                break
            end_idx = id + (len(segment_list) - offset)
            while not end_found and end_idx <= len(output_raw_map):
                restored_norm = " ".join([x[0] for x in output_raw_map[last_found_start_idx:][id: end_idx]])
                restored_raw = " ".join([x[1] for x in output_raw_map[last_found_start_idx:][id: end_idx]])
                # print("norm:", restored_norm)
                # print("raw :", restored_raw)
                # print("segm:", segment)

                processed_raw = process(normalizer.normalize(restored_raw).lower())
                processed_segment = process(segment.lower())
                processed_restored = process(restored_norm.lower())
                if processed_restored == processed_segment or processed_raw == processed_segment:
                    end_found = True
                    last_found_start_idx = end_idx
                elif processed_segment.startswith(processed_restored) or processed_segment.startswith(processed_raw):
                    end_idx += 1
                else:
                    restored_raw = NA
                    break

        restored.append(restored_raw)
    print(f'Restoration {key}: {round((perf_counter() - start_time), 2)} sec.')
    print(f"done with {key}")
    return (key, restored)

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
                    all_results[key] = process_file(item, key)[1]
                    not_found = [(i, r) for i, r in enumerate(all_results[key]) if r == NA]
                    num_not_found = len(not_found)
                    f.write(f'{key} -- number not found: {num_not_found} out of {len(all_results[key])} ({round(num_not_found/len(all_results[key])*100, 1)}%)')
                    f.write("\n")

                    if verbose:
                        print("=" * 40)
                        print(f'{key} -- number not found: {num_not_found} out of {len(all_results[key])} ({round(num_not_found / len(all_results[key]) * 100, 1)}%)')

                        if num_not_found > 0:
                            for idx, _ in not_found:
                                print(data[key]["segmented"][idx])
                        print("=" * 40)
                except Exception as e:
                    print(f"{key} -- FAILED -- {e}")
                    import pdb; pdb.set_trace()
                    f.write(f"{key} -- FAILED -- {e}")
                    f.write("\n")
                    raise e

    import pdb;

    pdb.set_trace()
    print()

    # results = Parallel(n_jobs=16)(delayed(process_file)(item, key) for key, item in tqdm(data.items()))

    print("DONE with parallel")

    import pdb; pdb.set_trace()
    print()