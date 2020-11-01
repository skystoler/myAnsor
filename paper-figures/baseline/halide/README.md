# Build Halide

### Install Dependency
1. Install ccache, libpng, libjpeg, llvm, clang
```
sudo apt update
sudo apt install libpng-dev libjpeg-dev

# install llvm, clang
...
```

2. add llvm-config, clang to the path. (In my test, llvm-9.0 works but llvm-8.0 raises some compiler errors)
```
sudo ln -s /usr/bin/llvm-config-9 /usr/bin/llvm-config
sudo ln -s /usr/bin/clang-9 /usr/bin/clang
sudo ln -s /usr/bin/clang++-9 /usr/bin/clang++
```

### Build Our Evaluation Fork
Our fork is necessary as we added new benchmark operators.

```
git clone --recursive git@github.com:merrymercy/Halide.git
cd Halide
git checkout ansor_eval
make -j32
cd apps/autoscheduler
make autotune -j32
```

# Run benchmark scripts
Set halide home to the root of your halide repo.
```
export HALIDE_HOME=~/projects/Halide
```

Run benchmark scripts
```
python3 benchmark.py
```

The results will be stored at `results.tsv`.
Then you can append it to our baseline collection file `paper-figures/baseline/results.tsv`
