import tvm
from tvm import te, ansor

from test_ansor_common import (matmul_ansor_test, conv2d_nchw_bn_relu_ansor_test,
                               max_pool2d_ansor_test, min_nm_ansor_test,
                               softmax_nm_ansor_test, softmax_abcd_ansor_test,
                               conv2d_winograd_nhwc_ansor_test)

def print_sketches(sketches):
    for i, s in enumerate(sketches):
        print("=" * 20 + " %d " % i + "=" * 20)
        print(s)

def generate_sketches(workload_func, args, target):
    workload_key = ansor.make_workload_key_func(workload_func, args)
    dag = ansor.workload_key_to_dag(workload_key)
    task = ansor.SearchTask(dag, workload_key, tvm.target.create(target))
    policy = ansor.SketchSearchPolicy(ansor.RandomModel())
    return policy.generate_sketches(task)

def test_cpu_matmul_sketch():
    sketches = generate_sketches(matmul_ansor_test, (512, 512, 512), 'llvm')
    ''' 3 multi-level tiling sketches
        0 - Multi-level tiling
        1 - Multi-level tiling with cache write on position 0
        2 - Multi-level tiling with cache write on position 1
    '''
    assert len(sketches) == 3

    sketches = generate_sketches(matmul_ansor_test, (8, 8, 512), 'llvm')
    ''' 2 rfactor sketches + 3 multi-level tiling sketches
        0 - Rfactor with factor position 0
        1 - Rfactor with factor position 1
        2 - Multi-level tiling
        3 - Multi-level tiling with cache write on position 0
        4 - Multi-level tiling with cache write on position 1
    '''
    assert len(sketches) == 5

def test_cpu_conv2d_bn_relu_sketch():
    sketches = generate_sketches(conv2d_nchw_bn_relu_ansor_test,
                                 (1, 56, 56, 512, 512, 3, 1, 1), 'llvm')
    ''' 3 multi-level tiling sketches
        0 - Conv2d multi-level tiling with fusion on position 0
        1 - Conv2d multi-level tiling with fusion on position 1
        2 - Conv2d multi-level tiling without fusion
    '''
    assert len(sketches) == 3

def test_cpu_max_pool2d_sketch():
    sketches = generate_sketches(max_pool2d_ansor_test, (1, 56, 56, 512, 1), 'llvm')
    assert len(sketches) == 1  # 1 valina sketch

def test_cpu_min_sketch():
    sketches = generate_sketches(min_nm_ansor_test, (10, 1024), 'llvm')
    assert len(sketches) == 3
    ''' 2 rfactor sketches + 1 default sketch
        0 - Rfactor with factor position 0
        1 - Rfactor with factor position 1
        2 - Default sketch
    '''

def test_cpu_softmax_sketch():
    sketches = generate_sketches(softmax_nm_ansor_test, (1, 1024), 'llvm')
    ''' (2 rfactor sketches + 1 default sketch) * (2 rfactor sketches + 1 default sketch) '''
    assert len(sketches) == 3 * 3

    sketches = generate_sketches(softmax_abcd_ansor_test, (1, 12, 128, 128), 'llvm')
    ''' (2 rfactor sketches + 1 default sketch) * (2 rfactor sketches + 1 default sketch) '''
    assert len(sketches) == 3 * 3

def test_cpu_conv2d_winograd_sketch():
    sketches = generate_sketches(conv2d_winograd_nhwc_ansor_test,
                                 (1, 28, 28, 128, 128, 3, 1, 1), 'llvm')
    ''' 3 multi-level tiling sketches
        0 - Bgemm multi-level tiling
        1 - Bgemm multi-level tiling with cache write on position 0
        2 - Bgemm multi-level tiling with cache write on position 1
    '''
    assert len(sketches) == 3

def test_gpu_matmul_sketch():
    if not tvm.context("cuda", 0).exist:
        return
    sketches = generate_sketches(matmul_ansor_test, (512, 512, 512), 'cuda')
    assert len(sketches) == 1  # 1 multi-level tiling

    sketches = generate_sketches(matmul_ansor_test, (8, 8, 1024), 'cuda')
    assert len(sketches) == 2  # 1 multi-level tiling + 1 cross thread reuction

def test_gpu_conv2d_bn_relu_sketch():
    if not tvm.context("cuda", 0).exist:
        return

    sketches = generate_sketches(conv2d_nchw_bn_relu_ansor_test,
                                 (1, 56, 56, 512, 512, 3, 1, 1), 'cuda')
    assert len(sketches) == 1  # 1 multi-level tiling sketches

def test_gpu_max_pool2d_sketch():
    if not tvm.context("cuda", 0).exist:
        return

    sketches = generate_sketches(max_pool2d_ansor_test, (1, 56, 56, 512, 0), 'cuda')
    assert len(sketches) == 1  # 1 valina sketch

def test_gpu_min_sketch():
    if not tvm.context("cuda", 0).exist:
        return

    sketches = generate_sketches(min_nm_ansor_test, (10, 1024), 'cuda')
    assert len(sketches) == 2  # 1 valina sketch + 1 cross-thread reduction

def test_gpu_softmax_sketch():
    if not tvm.context("cuda", 0).exist:
        return

    sketches = generate_sketches(softmax_nm_ansor_test, (2, 1024), 'cuda')
    # (1 valida sketch + 1 cross-thread reduction) ^ 2
    assert len(sketches) == 4

    sketches = generate_sketches(softmax_abcd_ansor_test, (1, 12, 128, 128), 'cuda')
    # (1 valida sketch + 1 cross-thread reduction) ^ 2
    assert len(sketches) == 4

def test_gpu_conv2d_winograd_sketch():
    sketches = generate_sketches(conv2d_winograd_nhwc_ansor_test,
                                 (1, 28, 28, 128, 128, 3, 1, 1), 'cuda')
    assert len(sketches) == 1

if __name__ == "__main__":
    test_cpu_matmul_sketch()
    test_cpu_conv2d_bn_relu_sketch()
    test_cpu_max_pool2d_sketch()
    test_cpu_min_sketch()
    test_cpu_softmax_sketch()
    test_cpu_conv2d_winograd_sketch()

    test_gpu_matmul_sketch()
    test_gpu_conv2d_bn_relu_sketch()
    test_gpu_max_pool2d_sketch()
    test_gpu_min_sketch()
    test_gpu_softmax_sketch()
    test_gpu_conv2d_winograd_sketch()

