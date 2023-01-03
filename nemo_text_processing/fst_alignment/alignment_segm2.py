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

# cache_dir = "/home/ebakhturina/NeMo/nemo_text_processing/text_normalization/cache_dir"
from nemo.utils import logging

NA = "n/a"


def find_segment(segment_id: int, output_raw_map, segmented, last_found_start_idx, offset: int=5):
    """
    Finds segment with punctuation in capitalization

    segment_id:
    output_raw_map:
    segmented:
    last_found_start_idx:

    Returns:
    """
    restored_raw = None
    norm_text = " ".join([x[0] for x in output_raw_map])
    norm_text_id = " ".join([x[0] + f"_{i}" for i, x in enumerate(output_raw_map)])
    raw_text = "|".join([x[1] for x in output_raw_map])
    cur_segment = segmented[segment_id].split()

    if len(cur_segment) == 0:
        return restored_raw, last_found_start_idx

    end_match_found = False

    try:
        pattern = cur_segment[0]
    except:
        import pdb; pdb.set_trace()
        print()
        print()
    max_start_match_len = min(4, len(cur_segment))
    for i in range(1, max_start_match_len):
        pattern += f"[^A-Za-z]+{cur_segment[i]}"

    pattern = re.compile(pattern)
    # if key == 'Am4BKyvYBgY' and segment_id == 177:
    #     [print(x) for x in pattern.finditer(norm_text)]
    #     import pdb; pdb.set_trace()
    #     print()

    for i, m in enumerate(pattern.finditer(norm_text.lower()[last_found_start_idx:])):
        if end_match_found:
            break

        match_idx = m.start() + last_found_start_idx
        stop_end_search = False
        end_idx = len(cur_segment) - offset
        norm_text_list = norm_text[match_idx:].split()

        while not end_match_found and end_idx <= len(norm_text_list) and not stop_end_search:
            restored = " ".join(norm_text_list[:end_idx])
            print(restored)
            import pdb;
            pdb.set_trace()
            if process(restored.lower()) == process(segmented[segment_id].lower()):
                stop_end_search = True
                end_match_found = True
                restored_raw = " ".join(raw_text[match_idx:].split()[:end_idx])
                last_found_start_idx = end_idx
            else:
                end_idx += 1
                if end_idx > (len(cur_segment) + offset):
                    stop_end_search = True
    return restored_raw, last_found_start_idx

def process(text):
    text = (text.replace(":", " ")
            .replace("-", " ")
            .replace(" .", ".")
            .replace(" ,", ",")
            .replace(" ?", "?"))

    text = text.replace("  ", " ")
    return text

def process_file(item, key, offset=5):
    print(f"processing {key}")
    restored = []
    text = item["raw_text"]

    if "segmented" not in item:
        return (key, restored)

    segmented = item["segmented"]

    if os.path.exists(f"alignment_{key}.p"):
        alignment = pickle.load(open(f"alignment_{key}.p", "rb"))
        output_text = pickle.load(open(f"output_text_{key}.p", "rb"))
    else:
        alignment, output_text = get_string_alignment(fst=fst, input_text=text, symbol_table=table)
        pickle.dump(alignment, open(f"alignment_{key}.p", "wb"))
        pickle.dump(output_text, open(f"output_text_{key}.p", "wb" ))

    restored = []
    indices = get_word_segments(text)
    output_raw_map = []
    for i, x in enumerate(indices):
        try:
            start, end = indexed_map_to_output(start=x[0], end=x[1], alignment=alignment)
        except:
            print(f"{key} -- error")
            return (key, restored)
            # import pdb; pdb.set_trace()
            # print()

        output_raw_map.append([output_text[start:end], text[x[0]:x[1]]])
    #     if i < 20:
    #         norm = output_text[start:end]
    #         print(f"{i} -- |{text[x[0]:x[1]]}| -- |{norm}|")
    # import pdb; pdb.set_trace()
    # print()

    # if os.path.exists(f"results_{key}.p"):
    #     results = pickle.load(open(f"results_{key}.p", "rb"))
    # else:
    #     results = Parallel(n_jobs=16)(delayed(find_segment)(segment_id, output_raw_map, segmented) for segment_id in range(0, len(segmented)))
    #     pickle.dump(alignment, open(f"results_{key}.p", "wb"))


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

                # if segment_id in [14, 16, 24, 38, 42, 79, 85, 89, 91, 95, 101, 106, 114, 136, 141, 145, 151, 158, 165, 168, 172, 178, 179, 181, 186, 187, 212, 225, 228] and len(segment_list) > 5:
                #     print("norm:", restored_norm)
                #     print("raw :", restored_raw)
                #     print("segm:", segment)
                #     import pdb; pdb.set_trace()
                #     print()


                if process(restored_norm.lower()) == process(segment.lower()):
                    end_found = True
                    last_found_start_idx = end_idx
                elif process(segment.lower()).startswith(process(restored_norm.lower())):
                    end_idx += 1
                else:
                    break

        restored.append(restored_raw)
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
    print(f"removing {audio_data_to_del} samples")
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
            if key == "AFDkXSGI2qA" or True:
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

    data = {"AFDkXSGI2qA": data["AFDkXSGI2qA"]}
    results = Parallel(n_jobs=16)(delayed(process_file)(item, key) for key, item in tqdm(data.items()))

    print("DONE with parallel")

    import pdb; pdb.set_trace()
    print()