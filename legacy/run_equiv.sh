#!/usr/bin/env bash
# Equivalence check: original clue-lds scripts vs chunked pipeline on the
# first 3M events ($EQUIV/sample.json), seed 1, 3 pairs. Needs Landauer's original
# scripts (github.com/ait-aecid/clue-lds) copied to $EQUIV/orig.
# The reference copy gets only two changes, neither affecting results:
#   randint(int(..)) for Python 3.12, and write_every 1e6 -> 1e3 (the
#   module-level `out_string +=` is quadratic on 3.12).
set -e
P="${PY:-python3}"
B="$(cd "$(dirname "$0")" && pwd)"
E="${EQUIV:-$B/../work/equiv}"
cd "$E/orig"
if [ ! -s det.log ]; then  # reference outputs are reused once produced
  sed -i "s/^write_every = 1000000/write_every = 1000/" generate_test_file.py
  grep -q "randint(int(earliest_change)" generate_test_file.py
  grep -q "^write_every = 1000 " generate_test_file.py
  $P get_user_similarity.py > sim.log 2>&1
  $P generate_test_file.py -s 1 -p 3 > gen.log 2>&1
  $P detect.py -t 0.6 > det.log 2>&1
fi
rm -rf "$E/chunked"
cd "$B"
$P stats.py --data "$E/sample.json" --out "$E/user_stats.json" > /dev/null
$P inject_split.py --data "$E/sample.json" --stats "$E/user_stats.json" --outdir "$E/chunked" --chunks 4 -s 1 -p 3 > "$E/inject.log"
for k in 0 1 2 3; do
  $P detect_chunk.py -t 0.6 --input "$E/chunked/chunk_$k.json" --labels "$E/chunked/labels.txt" --result "$E/chunked/result_$k.json" > /dev/null
done
$P merge.py "$E/chunked" > "$E/merged.txt"
echo DONE
