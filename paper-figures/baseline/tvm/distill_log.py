"""Distill a log file: Save the best logs from the original file into a new file

Usage:
python3 distill_log.py log.json
"""
import argparse
import numpy as np
import json

def key(data):
    res = ""
    res += data['i'][0]
    res += data['i'][1]
    res += str(data['i'][2][0])
    res += str(data['i'][2][1])
    res += str(data['i'][2][2])
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_file", type=str)
    parser.add_argument("--out-file", type=str, default=None)
    parser.add_argument("--remeasure", action='store_true', help='Replay the log to get more accurate measurement')
    args = parser.parse_args()

    if args.out_file is None:
        out_file = args.log_file
        out_file = out_file.replace(".log", ".best.log")
    else:
        out_file = args.out_file

    print("Loading the log file...")
    data_dict = {}
    count = 0
    with open(args.log_file, "r") as f:
        line = f.readline()
        while line:
            count += 1
            data = json.loads(line)
            if data['r'][1] == 0:
                if key(data) in data_dict.keys():
                    if np.mean(data['r'][0]) < np.mean(data_dict[key(data)]['r'][0]):
                        data_dict[key(data)] = data
                else:
                    data_dict[key(data)] = data
            line = f.readline()
    print("%d records processed" % count)

    with open(out_file, "w") as f:
        for data in data_dict:
            res = json.dumps(data_dict[data])
            f.write(res + "\n")

    print("The best records are written to output file %s" % out_file)

