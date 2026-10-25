"""Build detect_chunk.py from the original clue-lds detect.py.

Only I/O and parsing are patched (input/labels paths, result output,
orjson + fromisoformat); the detection logic and the TP/FP/TN/FN
bookkeeping are left byte-identical. Each patch must
match exactly once, so a changed upstream file fails loudly.

Chunking by user is exact because, in mode 1 (default) and 3, every
decision depends only on that user's own events. Mode 2 (idf) counts users
across the whole file and is therefore refused.
"""
import sys

src_path, dst_path = sys.argv[1], sys.argv[2]
src = open(src_path, encoding='utf-8').read()

patches = [
    ("argparser.add_argument('-d','--debug'",
     "argparser.add_argument('--input', required=True)\n"
     "argparser.add_argument('--labels', required=True)\n"
     "argparser.add_argument('--result', required=True)\n"
     "argparser.add_argument('-d','--debug'"),
    ("fifo = args[\"fifo\"]\n",
     "fifo = args[\"fifo\"]\n"
     "if mode == 2:\n"
     "    raise SystemExit('mode 2 (idf) is not chunk-safe')\n"),
    ("with open('labels.txt') as f:", "with open(args['labels']) as f:"),
    ("with open('clue_anomaly.json') as f:", "with open(args['input'], 'rb') as f:"),
    # Speed: orjson parses bytes ~3x faster than json; fromisoformat gives the
    # same datetime fields as dateutil isoparse for "YYYY-MM-DDTHH:MM:SSZ".
    ("import json\n", "import json\nimport orjson\n"),
    ("        j = json.loads(line)\n", "        j = orjson.loads(line)\n"),
    ("        ts = parser.isoparse(j['time'])\n", "        ts = datetime.datetime.fromisoformat(j['time'])\n"),
    ("print('Results with threshold = ' + str(threshold) + ':')\nget_eval_results(detected_dist)\n", ""),
]
for old, new in patches:
    n = src.count(old)
    if n != 1:
        raise SystemExit(f'patch anchor found {n}x: {old!r}')
    src = src.replace(old, new)

# Same classification rules as get_eval_results(), but emitting raw counts
# so chunks can be summed by merge.py.
src += '''

def dump_counts(d):
    c = dict(tp=0, tp_adj=0, fp=0, tn=0, fn=0, fn_adj=0, train=0, detect=0)
    per_user = {}
    for tup, detected in d.items():
        if "training" in detected:
            c['train'] += 1
        elif "detection" in detected:
            c['detect'] += 1
        uid, day = tup
        if uid in anomalous_users and day.timestamp() == anomalous_users[uid].timestamp():
            pu = per_user.setdefault(uid, dict(tp_adj=0, fn=0))
            if detected == ["detection", "anomalous"]:
                c['tp'] += 1; c['tp_adj'] += 1; pu['tp_adj'] += 1
            elif detected == ["detection", "normal"] or detected == ["training", "normal"]:
                c['fn_adj'] += 1; c['fn'] += 1; pu['fn'] += 1
            elif detected == ["training", "anomalous"]:
                c['tp_adj'] += 1; pu['tp_adj'] += 1; c['fn'] += 1
        else:
            if detected == ["detection", "anomalous"]:
                c['fp'] += 1
            elif detected == ["detection", "normal"]:
                c['tn'] += 1
    c['per_user'] = per_user
    c['users'] = len(total_days)
    c['user_days'] = sum(len(v) for v in total_days.values())
    c['runtime'] = time.time() - start_time
    c['params'] = dict(threshold=threshold, retrain=anom_free_days, mode=mode, queue=queue, update=update, fifo=fifo)
    with open(args['result'], 'w') as out:
        json.dump(c, out)
    print('\\nchunk done: ' + str({k: v for k, v in c.items() if k not in ('per_user', 'params')}))

dump_counts(detected_dist)
'''
open(dst_path, 'w', encoding='utf-8').write(src)
print('wrote', dst_path)
