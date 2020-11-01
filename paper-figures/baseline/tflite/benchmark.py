import argparse
import time
import tvm
import numpy as np
from tvm.contrib import util, tflite_runtime
from pathlib import Path
from tvm.auto_scheduler.utils import request_remote
from tensorflow import lite as interpreter_wrapper
from utils import log_line, BenchmarkRecord

def run_tflite(device_key, host, port, num_threads, tflite_model_path, timeout):
    # inference via tflite interpreter python apis
    interpreter = interpreter_wrapper.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    tflite_input = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], tflite_input)
    interpreter.invoke()
    tflite_output = interpreter.get_tensor(output_details[0]['index'])
    remote = request_remote(device_key, host, port, timeout=timeout)
    ctx = remote.cpu(0)
    remote.upload(tflite_model_path)
    # inference via tvm tflite runtime
    with open(tflite_model_path, 'rb') as model_fin:
        runtime = tflite_runtime.create(model_fin.read(), ctx=ctx)
        runtime.set_input(0, tvm.nd.array(tflite_input, ctx=ctx))
        runtime.set_num_threads(num_threads)
        runtime.invoke()
        out = runtime.get_output(0)
        np.testing.assert_allclose(out.asnumpy(), tflite_output, atol=1e-5, rtol=1e-5)
        print("Evaluate inference time cost...")
        ftimer = runtime.module.time_evaluator("invoke", ctx, number=100, repeat=3)
        prof_res = np.array(ftimer().results) * 1000
        print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
              (np.mean(prof_res), np.std(prof_res)))
        return np.mean(np.array(ftimer().results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='aarch64-Cortex-A53-1.4Ghz')
    parser.add_argument("--tflite-model-path", type=str, required=True)
    parser.add_argument("--target", type=str, default='llvm -device=arm_cpu -model=rasp3b+ '
                                                      '-target=aarch64-linux-gnu -mcpu=cortex-a53 -mattr=+neon')
    parser.add_argument("--target-host", type=str, default=None)
    parser.add_argument("--device-key", type=str, default='rasp')
    parser.add_argument("--host", type=str, default='0.0.0.0')
    parser.add_argument("--port", type=int, default=9190)
    parser.add_argument("--num-threads", type=int, required=True)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--out-file", type=str, default='results.tsv')

    args = parser.parse_args()
    cost = run_tflite(args.device_key, args.host, args.port, args.num_threads, args.tflite_model_path, args.timeout)

    log_line(BenchmarkRecord(args.device, 'cpu', 'network', Path(args.tflite_model_path).stem,
                             'tflite', 'tflite-' + str(args.num_threads) + "-core(s)", {"costs": cost},
                             time.time()), args.out_file)

