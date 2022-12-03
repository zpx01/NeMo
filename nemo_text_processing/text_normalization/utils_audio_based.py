from cdifflib import CSequenceMatcher

from nemo.utils import logging

logging.setLevel("INFO")

MATCH = "match"
NONMATCH = "non-match"
SEMIOTIC_TAG = "[SEMIOTIC_SPAN]"


def _get_alignment(a, b):
    """

	TODO: add

	Returns:
		list of Tuple(pred start and end, gt start and end) subsections
	"""
    a = a.lower().split()
    b = b.lower().split()

    s = CSequenceMatcher(None, a, b, autojunk=False)
    # s contains a list of triples. Each triple is of the form (i, j, n), and means that a[i:i+n] == b[j:j+n].
    # The triples are monotonically increasing in i and in j.
    s = s.get_matching_blocks()

    diffs = {}
    non_match_start_l = 0
    non_match_start_r = 0
    for match in s:
        l_start, r_start, length = match
        if non_match_start_l < l_start:
            while non_match_start_l < l_start:
                diffs[non_match_start_l] = (non_match_start_r, r_start, NONMATCH)
                non_match_start_l += 1

        for len_ in range(length):
            diffs[l_start + len_] = (r_start + len_, r_start + 1 + len_, MATCH)
        non_match_start_l = l_start + length
        non_match_start_r = r_start + length

    # for k, v in diffs.items():
    #     # import pdb; pdb.set_trace()
    #     if "Q" in a[k] or True:
    #         print(f"{k}: {a[k]} -- {b[v[0]:v[1]]} -- {v[2]}")
    #         #
    #         # print()
    # import pdb;
    # pdb.set_trace()
    return diffs


def get_semiotic_spans(a, b):
    """returns list of different substrings between and b

	Returns:
		list of Tuple(pred start and end, gt start and end) subsections
	"""

    b = b.lower().split()
    s = CSequenceMatcher(None, a.lower().split(), b, autojunk=False)

    # do it here to preserve casing
    a = a.split()

    result = []
    # s contains a list of triples. Each triple is of the form (i, j, n), and means that a[i:i+n] == b[j:j+n].
    # The triples are monotonically increasing in i and in j.
    a_start_non_match, b_start_non_match = 0, 0
    # get not matching blocks
    for match in s.get_matching_blocks():
        a_start_match, b_start_match, match_len = match
        # we're widening the semiotic span to include 1 context word from each side, so not considering 1-word match here
        if match_len > 1:
            if a_start_non_match < a_start_match:
                result.append([[a_start_non_match, a_start_match], [b_start_non_match, b_start_match]])
            a_start_non_match = a_start_match + match_len
            b_start_non_match = b_start_match + match_len

    if a_start_non_match < len(a):
        result.append([[a_start_non_match, len(a)], [b_start_non_match, len(b)]])

    # add context (1 word from both sides)
    result_with_context = []
    for item in result:
        try:
            start_a, end_a = item[0]
            start_b, end_b = item[1]
        except:
            import pdb

            pdb.set_trace()
        if start_a - 1 >= 0 and start_b - 1 >= 0:
            start_a -= 1
            start_b -= 1
        if end_a + 1 <= len(a) and end_b + 1 <= len(b):
            end_a += 1
            end_b += 1
        result_with_context.append([[start_a, end_a], [start_b, end_b]])

    result = result_with_context
    semiotic_spans = []
    default_normalization = []
    a_masked = []
    start_idx = 0
    masked_idx = []
    for item in result:
        cur_start, cur_end = item[0]
        if start_idx < cur_start:
            a_masked.extend(a[start_idx:cur_start])
        start_idx = cur_end
        a_masked.append(SEMIOTIC_TAG)
        masked_idx.append(len(a_masked) - 1)

    if start_idx < len(a):
        a_masked.extend(a[start_idx:])

    for item in result:
        a_diff = ' '.join(a[item[0][0] : item[0][1]])
        b_diff = ' '.join(b[item[1][0] : item[1][1]])
        semiotic_spans.append(a_diff)
        default_normalization.append(b_diff)
        logging.debug(f"a: {a_diff}")
        logging.debug(f"b: {b_diff}")
        logging.debug("=" * 20)
    return result, semiotic_spans, a_masked, masked_idx, default_normalization


def _print_alignment(diffs, l, r):
    l = l.split()
    r = r.split()
    for l_idx, item in diffs.items():
        start, end, match_state = item
        print(f"{l[l_idx]} -- {r[start: end]} -- {match_state}")


def _adjust_span(semiotic_spans, norm_pred_diffs, pred_norm_diffs, norm_raw_diffs, raw: str, pred_text: str):
    """

	:param semiotic_spans:
	:param norm_pred_diffs:
	:param pred_norm_diffs:
	:param norm_raw_diffs:
	:param raw:
	:param pred_text:
	:return:
	"""
    standard_start = 0
    raw_text_list = raw.split()
    pred_text_list = pred_text.split()

    text_for_audio_based = {"semiotic": [], "standard": "", "pred_text": []}

    for idx, (raw_span, norm_span) in enumerate(semiotic_spans):
        raw_start, raw_end = raw_span
        raw_end -= 1

        # get the start of the span
        pred_text_start, _, status_norm_pred = norm_pred_diffs[norm_span[0]]
        new_norm_start = pred_norm_diffs[pred_text_start][0]
        raw_text_start = norm_raw_diffs[new_norm_start][0]

        # get the end of the span
        _, pred_text_end, status_norm_pred = norm_pred_diffs[norm_span[1] - 1]

        new_norm_end = pred_norm_diffs[pred_text_end - 1][1]
        raw_text_end = norm_raw_diffs[new_norm_end - 1][1]

        if standard_start < raw_text_start:
            text_for_audio_based["standard"] += " " + " ".join(raw_text_list[standard_start:raw_text_start])

        cur_semiotic_span = f"{' '.join(raw_text_list[raw_text_start:raw_text_end])}"
        cur_pred_text = f"{' '.join(pred_text_list[pred_text_start:pred_text_end])}"

        text_for_audio_based["semiotic"].append(cur_semiotic_span)
        text_for_audio_based["pred_text"].append(cur_pred_text)
        text_for_audio_based["standard"] += f" {SEMIOTIC_TAG} "

        standard_start = raw_text_end

    if standard_start < len(raw_text_list):
        text_for_audio_based["standard"] += ' '.join(raw_text_list[standard_start:])

    text_for_audio_based["standard"] = text_for_audio_based["standard"].replace("  ", " ").strip()
    return text_for_audio_based


def adjust_boundaries(norm_raw_diffs, norm_pred_diffs, raw, norm, pred_text):
    adjusted = []
    word_id = 0
    while word_id < len(norm.split()):
        norm_raw, norm_pred = norm_raw_diffs[word_id], norm_pred_diffs[word_id]
        if (norm_raw[2] == MATCH and norm_pred[2] == NONMATCH) or (norm_raw[2] == NONMATCH and norm_pred[2] == MATCH):
            mismatched_id = word_id
            non_match_raw_start = norm_raw[0]
            non_match_pred_start = norm_pred[0]
            done = False
            word_id += 1
            while word_id < len(norm.split()) and not done:
                norm_raw, norm_pred = norm_raw_diffs[word_id], norm_pred_diffs[word_id]
                if norm_raw[2] == MATCH and norm_pred[2] == MATCH:
                    non_match_raw_end = norm_raw_diffs[word_id - 1][1]
                    non_match_pred_end = norm_pred_diffs[word_id - 1][1]
                    word_id -= 1
                    done = True
                else:
                    word_id += 1
            if not done:
                non_match_raw_end = len(raw.split())
                non_match_pred_end = len(pred_text.split())
            adjusted.append(
                (
                    mismatched_id,
                    (non_match_raw_start, non_match_raw_end, NONMATCH),
                    (non_match_pred_start, non_match_pred_end, NONMATCH),
                )
            )
        else:
            # print(word_id, "2")
            adjusted.append((word_id, norm_raw, norm_pred))
        word_id += 1

    adjusted2 = []
    last_status = None
    for idx, item in enumerate(adjusted):
        if last_status is None:
            last_status = item[1][2]
            raw_start = item[1][0]
            pred_text_start = item[2][0]
            norm_span_start = item[0]
            raw_end = item[1][1]
            pred_text_end = item[2][1]
        elif last_status is not None and last_status == item[1][2]:
            raw_end = item[1][1]
            pred_text_end = item[2][1]
        else:
            adjusted2.append(
                ([norm_span_start, item[0]], [raw_start, raw_end], [pred_text_start, pred_text_end], last_status)
            )
            last_status = item[1][2]
            raw_start = item[1][0]
            pred_text_start = item[2][0]
            norm_span_start = item[0]
            raw_end = item[1][1]
            pred_text_end = item[2][1]

    if last_status == item[1][2]:
        raw_end = item[1][1]
        pred_text_end = item[2][1]
        adjusted2.append(
            ([norm_span_start, item[0]], [raw_start, raw_end], [pred_text_start, pred_text_end], last_status)
        )
    else:
        adjusted2.append(
            (
                [adjusted[idx - 1][0], len(norm.split())],
                [item[1][0], len(raw.split())],
                [item[2][0], len(pred_text.split())],
                item[1][2],
            )
        )

    # print("+" * 50)
    # print("adjusted:")
    # for item in adjusted2:
    #     # import pdb; pdb.set_trace()
    #     print(f"{raw.split()[item[1][0]: item[1][1]]} -- {pred_text.split()[item[2][0]: item[2][1]]}")
    #     # print(item)
    #
    # print("+" * 50)
    # print("adjusted2:")
    # for item in adjusted2:
    #     print(f"{raw.split()[item[1][0]: item[1][1]]} -- {pred_text.split()[item[2][0]: item[2][1]]}")
    # print("+" * 50)
    raw_list = raw.split()
    pred_text_list = pred_text.split()
    norm_list = norm.split()

    extended_spans = []
    adjusted3 = []
    idx = 0
    while idx < len(adjusted2):
        item = adjusted2[idx]
        cur_semiotic = " ".join(raw_list[item[1][0] : item[1][1]])
        cur_pred_text = " ".join(pred_text_list[item[2][0] : item[2][1]])
        cur_norm_span = " ".join(norm_list[item[0][0] : item[0][1]])

        if len(cur_pred_text) == 0:
            # print(f"current: {item}")
            # import pdb; pdb.set_trace()
            raw_start, raw_end = item[0]
            norm_start, norm_end = item[1]
            pred_start, pred_end = item[2]
            while idx < len(adjusted2) - 1 and not ((pred_end - pred_start) > 2 and adjusted2[idx][-1] == MATCH):
                idx += 1
                raw_end = adjusted2[idx][0][1]
                norm_end = adjusted2[idx][1][1]
                pred_end = adjusted2[idx][2][1]
                # print("="*40)
                # print(item)
                # print(f"raw : -- {cur_semiotic}")
                # print(f"pred: -- {cur_pred_text}")
                # print(f"norm: -- {cur_norm_span}")
            cur_item = ([raw_start, raw_end], [norm_start, norm_end], [pred_start, pred_end], NONMATCH)
            adjusted3.append(cur_item)
            extended_spans.append(len(adjusted3) - 1)
            idx += 1
        else:
            adjusted3.append(item)
            idx += 1

    semiotic_spans = []
    norm_spans = []
    pred_texts = []
    raw_text_masked = ""
    for idx, item in enumerate(adjusted3):
        cur_semiotic = " ".join(raw_list[item[1][0] : item[1][1]])
        cur_pred_text = " ".join(pred_text_list[item[2][0] : item[2][1]])
        cur_norm_span = " ".join(norm_list[item[0][0] : item[0][1]])
        # print("="*40)
        # print(item)
        # print(f"raw : -- {cur_semiotic}")
        # print(f"pred: -- {cur_pred_text}")
        # print(f"norm: -- {cur_norm_span}")

        if idx == len(adjusted3) - 1:
            cur_norm_span = " ".join(norm_list[item[0][0] : len(norm_list)])
        if (item[-1] == NONMATCH and cur_semiotic != cur_norm_span) or (idx in extended_spans):
            raw_text_masked += " " + SEMIOTIC_TAG
            semiotic_spans.append(cur_semiotic)
            pred_texts.append(cur_pred_text)
            norm_spans.append(cur_norm_span)
        else:
            raw_text_masked += " " + " ".join(raw_list[item[1][0] : item[1][1]])
    # print(len(adjusted2), len(adjusted3))
    # import pdb; pdb.set_trace()
    raw_text_masked_list = raw_text_masked.strip().split()

    raw_text_mask_idx = [idx for idx, x in enumerate(raw_text_masked_list) if x == SEMIOTIC_TAG]

    # print("+" * 50)
    # print("adjusted3:")
    # for item in adjusted3:
    #     # import pdb; pdb.set_trace()
    #     print(f"{raw.split()[item[1][0]: item[1][1]]} -- {pred_text.split()[item[2][0]: item[2][1]]}")
    #     # print(item)
    #
    # print("+" * 50)
    return semiotic_spans, pred_texts, norm_spans, raw_text_masked_list, raw_text_mask_idx


def get_alignment(raw, norm, pred_text, verbose: bool = False):
    import time

    # start_time = time.time()
    # semiotic_spans = get_semiotic_spans(raw, norm)[0]
    # print(f'Alignment 1: {round((time.time() - start_time) / 60, 2)} min.')

    norm_pred_diffs = _get_alignment(norm, pred_text)
    norm_raw_diffs = _get_alignment(norm, raw)
    # for i in range(len(norm.split())):
    #     print(i, norm_raw_diffs[i], norm_pred_diffs[i])

    semiotic_spans, pred_texts, norm_spans, raw_text_masked_list, raw_text_mask_idx = adjust_boundaries(
        norm_raw_diffs, norm_pred_diffs, raw, norm, pred_text
    )

    for i in range(len(semiotic_spans)):
        print("=" * 40)
        # print(i)
        print(f"semiotic : {semiotic_spans[i]}")
        print(f"pred text: {pred_texts[i]}")
        print(f"norm     : {norm_spans[i]}")
        print("=" * 40)
        # import pdb; pdb.set_trace()
    # print(" ".join(raw_text_masked_list))
    # import pdb; pdb.set_trace()
    # [print(i, x) for i, x in enumerate(semiotic_spans) if x == ""]
    # [print(i, x) for i, x in enumerate(pred_texts) if x == ""]
    return semiotic_spans, pred_texts, norm_spans, raw_text_masked_list, raw_text_mask_idx


if __name__ == "__main__":
    raw = "This, example: number 1,500 can be a very long one!, and can fail to produce valid normalization for such an easy number like 10,125 or dollar value $5349.01, and can fail to terminate, and can fail to terminate, and can fail to terminate, 452."
    norm = "This, example: number one thousand five hundred can be a very long one!, and can fail to produce valid normalization for such an easy number like ten thousand one hundred twenty five or dollar value five thousand three hundred and forty nine dollars and one cent, and can fail to terminate, and can fail to terminate, and can fail to terminate, four fifty two."
    pred_text = "this w example nuber viteen hundred can be a very h lowne one and can fail to produce a valid normalization for such an easy number like ten thousand one hundred twenty five or dollar value five thousand three hundred and fortyn nine dollars and one cent and can fail to terminate and can fail to terminate and can fail to terminate four fifty two"

    import json
    from time import perf_counter

    det_manifest = (
        f"/mnt/sdb/DATA/SPGI/normalization//sample_hour.json"  # deter TN predictions stored in "pred_text" field
    )
    with open(det_manifest, "r") as f:
        for line in f:
            line = json.loads(line)
            norm = line["deter_tn"]
            raw = line["text"]
            pred_text = line["pred_text"]

    # raw = "This is just the first step of a more ambitious long-term plan to open 1,300 stores in the next 10 years. and Fernando just showed you the phasing of the implementation plan on Slide 20. On Slide 21, we have translated the first 3 years of our transformation plan into a more detailed financial outlook. That will obviously take some time to achieve."
    # norm = "This is just the first step of a more ambitious long-term plan to open one thousand three hundred stores in the next ten years. and Fernando just showed you the phasing of the implementation plan on Slide twenty. On Slide twenty one, we have translated the first three years of our transformation plan into a more detailed financial outlook. That will obviously take some time to achieve."
    # pred_text = "he just the firstastate of a more ambitious long term plan to oten thirteen hundred stores in the next ten years and tonata just shows you the phasing of the implementation plan on slyg twenty of flight twenty one we have translated the first three years of our transformation clamb into a more detailed financial outlook am that will obsesly take some time to achieve"

    # raw = "We've got a lot of work ahead of us in Q2 and Q3. But we feel like that we started down the right path."
    # norm = "We've got a lot of work ahead of us in Q two and Q3 three. But we feel like that we started down the right path."
    # pred_text = "we've got a lot of work ahead of us and q too and few three but we feel like that we've started down the right path"

    # raw = "Our guidance does not include the impact of future acquisitions. After taking on the additional operational oversights since the departure of our COO, and spending the past three months in the field working directly with our facility CEOs, I can tell you that we are making dramatic improvements across all areas of our clinical operations. had hoped it would. We've got a lot of work ahead of us in Q2 and Q3. But we feel like that we started down the right path."
    # norm = "Our guidance does not include the impact of future acquisitions. After taking on the additional operational oversights since the departure of our COO, and spending the past three months in the field working directly with our facility CEOs, I can tell you that we are making dramatic improvements across all areas of our clinical operations. had hoped it would. We've got a lot of work ahead of us in Q two and Q three. But we feel like that we started down the right path."
    # pred_text = "our guidance does not include the impact of future acquisitions after taking on the additional operational oversight since the departure of our c o o and spending the past three months in the field working directly with our facility c e os i can tell you that we are making dramatic improvements across all areas of our clinical operations had hoped it would we've got a lot of work ahead of us and q too and few three but we feel like that we've started down the right path"
    #
    raw = "of getting it done this year or at worst case very early into calendar year '19. So things are progressing well there. We also filed for the key state and federal approvals in 2017, and we'll go over that with a chart in a second. We also expect to file rate cases by May for both NYSEG and RG&E for electric and gas by May. The contracts now are expected to be approved by the second quarter of '19, and this just puts it in context with the time frame we expected all along. We received our FERC approval That was filed in September of 2017, and we should have that by mid-2019. And then local and municipal construction approvals will be timed as needed throughout the project. Good afternoon."
    norm = "of getting it done this year or at worst case very early into calendar year 'nineteen. So things are progressing well there. We also filed for the key state and federal approvals in twenty seventeen, and we'll go over that with a chart in a second. We also expect to file rate cases by May for both NYSEG and RG&E for electric and gas by May. The contracts now are expected to be approved by the second quarter of 'nineteen, and this just puts it in context with the time frame we expected all along. We received our FERC approval That was filed in September of twenty seventeen, and we should have that by mid- twenty nineteen. And then local and municipal construction approvals will be timed as needed throughout the project. Good afternoon."
    pred_text = "of of getting it done this year or at worst case on very early in to count the year nineteent so things are progressing wele there u we also filed for the key state and federal approvals in two thousand seventeen and will go over that with a chart in a second we also expect to file ratecases by may for both niig and argeni for electric and gas byma the contract now are expected be approved by the second quarter of nineteen and this is puts it in contact with the timeframe we expected all along we received our fork approval that was filed in september of twenty seventeen and we should have that bive id twenty nineteen and then local municipal construction approvals will be timed as needed throughout the project good afternoon"
    # with open("debug.json", "w") as f:
    #     item = {"text": raw, "deter_tn": norm, "pred_text": pred_text}
    #     f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # get_semiotic_spans(norm, pred_text)
    # print("\n\n")
    # get_semiotic_spans(pred_text, norm)
    # import pdb; pdb.set_trace()

    start_time = perf_counter()
    text_for_audio_based = get_alignment(raw, norm, pred_text, verbose=True)
    print(f'Execution time: {round((perf_counter() - start_time) / 60, 2)} min.')

    # print(text_for_audio_based)

    # raw = "We have spent the last several years reshaping our branch network, upgrading technology and deepening our focus on our core 6 markets,"
    # norm = "we have spent the last several years reshaping our branch network upgrading technology and deepening our focus on our core six markets,"
    #
    # result = get_semiotic_spans(raw, norm)
