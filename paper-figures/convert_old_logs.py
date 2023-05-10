"""Convert logs from v0.1 (tvm-autoscheduler repo) to v0.2 (ansor repo)"""

from common import run_cmd

def convert(filename):
    replace_pair = [
        ('"RS"', '"RE"'),
        ('"SS"', '"SP"'),
        ('"FSS"', '"FSP"'),
        ('"FFSS"', '"FFSP"'),
        ('"FS"', '"FU"'),
        ('"AS"', '"AN"'),
        ('"PS"', '"PR"'),
        ('"RFS"', '"RF"'),
    ]

    for before, after in replace_pair:
        run_cmd("""sed -i "" 's/%s/%s/g' %s""" % (before, after, filename))

if __name__ == "__main__":
    prefix = "saved_logs/2020-05-21-single-op-ablation"

    names = ['op-beam-single', 'op-full-single', 'op-limit-space-single', 'op-no-fine-tune-single']
    median = 5

    for name in names:
        for suffix in range(median):
            filename = "%s/%s.json.%d" % (prefix, name, suffix)
            convert(filename)

