 # Performance

Performance is critical to `dream_go` as any underutilized resouces could be better spent better understanding the current board position. But performance is a tricky question, especially when dealing with multi-threaded and multiple devices, luckily there are a few tools that are helpful.

First you may wish to enable debug symbols for the release build, as it will allow all of these tools to produce more human readable outputs. Do not commit this change as these symbols take up a fair amount of space, which we do not want to include in official releases.

```
[profile.release]
debug = true
```

## `perf`

The `perf` command makes use of the linux performance counters to collect profiling information. This set of tools has incredibly low overhead and produces a set of annotated assembly code.

```
perf record ./target/release/dream_go --bench
perf report
```

## `callgrind`

Part of the `valgrind` family of tools, an _emulator_ that can profile memory access and cache lines in addition to just collecting stack traces. This is very slow since `valgrind` has to collect all of the stack traces in user space, but provides very detailed call stacks:

```
valgrind --tool=callgrind --dump-instr=yes --collect-jumps=yes --simulate-cache=yes ./target/release/dream_go --bench
callgrind_annotate --inclusive=yes --tree=both callgrind.out.<pid>
```

## `nvprof`

For profiling CUDA the best tool is the ones provided by NVIDIA, in this case the `nvprof` tool keep track of kernel executions and time taken on both host and device set.

```
nvprof ./target/release/dream_go --bench
```
