---
title: CUDA执行模型：从Kernel到Warp调度
date: 2025-10-21 14:55:35
tags:
---

在传统的 CPU 编程中，我们使用线程（Thread）作为并行的基本单位。一个程序可能创建 8 、16 或 32 个线程来充分利用 CPU 的多个核心。

但这种模型在扩展到 GPU 时会遇到根本性的障碍。

<!--more-->

---

## 1. CPU 线程模型的困境

让我们量化 CPU 线程的成本。以 Linux 系统为例：

**单个 CPU 线程的开销**：

| 资源类型                       | 大小        | 说明                     |
| ------------------------------ | ----------- | ------------------------ |
| **内核栈**                     | 8 KB        | 用于系统调用和中断处理   |
| **用户栈**                     | 8 MB (默认) | 可通过 ulimit 调整       |
| **Thread Control Block (TCB)** | ~1 KB       | 线程状态、寄存器上下文等 |
| **线程局部存储 (TLS)**         | ~4 KB       | 每线程的全局变量副本     |
| **总计**                       | ~8 MB       | 每个线程约占 8MB 内存    |

如果我们想在 CPU 上创建 10,000 个线程（GPU 常见的并行度）：

```
内存需求 = 10,000 × 8 MB = 80 GB
```

我们还要考虑线程切换的开销。

**上下文切换开销**：

当操作系统从线程 A 切换到线程 B 时，需要执行以下步骤：

| 步骤               | 操作                                  | 延迟                        |
| ------------------ | ------------------------------------- | --------------------------- |
| 1. 保存线程 A 状态 | 保存 16-32 个通用寄存器 + PC + 标志位 | ~50 cycles                  |
| 2. TLB 刷新        | 清空虚拟地址转换缓存                  | ~100 cycles                 |
| 3. 缓存污染        | 线程 B 的数据驱逐线程 A 的缓存        | ~1000 cycles                |
| 4. 恢复线程 B 状态 | 从内存加载寄存器上下文                | ~50 cycles                  |
| **总计**           |                                       | **~1200 cycles ≈ 0.3 微秒** |

在一个 5 GHz 的 CPU 上，每次上下文切换损失约 0.3 微秒。如果频繁在 10,000 个线程间切换，开销会完全压垮系统。

**关键问题**：为什么 CPU 线程如此昂贵？

1. **独立地址空间**：每个线程有独立的虚拟内存映射，需要 TLB 支持
2. **抢占式调度**：操作系统可以在任意时刻打断线程，需要保存完整状态
3. **丰富的系统调用**：线程可以执行文件 I/O、网络访问等复杂操作

GPU 放弃了这些特性以换取轻量级的线程模型。

### GPU 需要的并行模型

深度学习的典型 workload 特征：

```python
# 矩阵乘法：C = A × B
# A: [4096, 4096], B: [4096, 4096], C: [4096, 4096]

for i in range(4096):
  for j in range(4096):
    sum = 0.0
    for k in range(4096):
      sum += A[i][k] * B[k][j]
      C[i][j] = sum
```

**并行度分析**：

- 外层循环：4096 × 4096 = 16,777,216 个独立的输出元素
- 每个元素的计算：4096 次乘加操作
- **完全独立**：计算 C[0][0]不需要 C[0][1]的结果

这种大规模数据并行（Data Parallelism）有以下特点：

1. **海量并行任务**：数百万个独立计算
2. **简单控制流**：没有复杂的 if-else、函数调用
3. **相似计算模式**：所有任务执行相同的代码，只是数据不同
4. **短生命周期**：每个任务执行几微秒就结束
5. **无需系统调用**：不访问文件、网络等

**GPU 线程模型的设计目标**：

| 特性         | CPU 线程     | GPU 线程        | 设计权衡           |
| ------------ | ------------ | --------------- | ------------------ |
| **创建开销** | 高 (~50 μs)  | 极低 (~0.01 μs) | GPU 无需系统调用   |
| **内存占用** | 8 MB/线程    | ~1 KB/线程      | GPU 无独立栈       |
| **切换开销** | 高 (~0.3 μs) | 零              | 使用硬件调度       |
| **最大数量** | 数千         | 数百万          | 面向大规模并行     |
| **调度方式** | 抢占式       | 协作式          | GPU 线程不能被中断 |
| **独立性**   | 完全独立     | 按组执行        | GPU 采用 SIMT      |

---

## 2. CUDA 线程层次结构：Grid-Block-Thread

### 三层抽象的设计哲学

CUDA 将并行计算组织为三个层次：**Grid → Block → Thread**。这不是任意的设计，而是精确映射到 GPU 的硬件结构。

```
软件抽象           硬件对应
┌─────────────┐   ┌──────────────┐
│   Grid      │ → │  整个GPU      │
│  (Kernel)   │   │  (所有SM)     │
└─────────────┘   └──────────────┘
      │                  │
      ├── Block 0  →  调度到 SM 0
      ├── Block 1  →  调度到 SM 1
      ├── Block 2  →  调度到 SM 0
      │   ...           ...
      └── Block N  →  调度到 SM x
            │                │
            ├── Thread 0     │
            ├── Thread 1     │  在SM内部
            │   ...          │  组成warp
            └── Thread M     │
```

### Kernel：GPU 上的"程序"

**Kernel 的定义**：

```cuda
// 声明一个kernel函数
__global__ void vectorAdd(float* A, float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

// 从CPU启动kernel
int N = 1024;
int threadsPerBlock = 256;
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
```

**Kernel 的关键特性**：

| 特性                   | 说明                          | 限制                             |
| ---------------------- | ----------------------------- | -------------------------------- |
| **`__global__`修饰符** | 表示这是 GPU 函数，CPU 可调用 | 返回值必须是 void                |
| **参数传递**           | 通过值传递，指针指向 GPU 内存 | 参数总大小 < 4 KB                |
| **<<<...>>>语法**      | 指定 Grid 和 Block 维度       | 最多 3 维                        |
| **异步执行**           | kernel 启动后 CPU 继续执行    | 需要 cudaDeviceSynchronize()同步 |
| **无递归（旧架构）**   | Fermi 之前不支持递归          | Kepler 及以后支持动态并行        |

### Grid：整个 Kernel 的执行空间

Grid 是 kernel 的最外层容器，定义了总的并行度。

**Grid 的维度配置**：

```cuda
// 1D Grid
kernel<<<numBlocks, threadsPerBlock>>>(...)
// 等价于
dim3 grid(numBlocks);
dim3 block(threadsPerBlock);
kernel<<<grid, block>>>(...)

// 2D Grid（适合2D数据，如图像）
dim3 grid(gridWidth, gridHeight);
dim3 block(blockWidth, blockHeight);
kernel<<<grid, block>>>(...)

// 3D Grid（适合3D数据，如体积数据）
dim3 grid(gridX, gridY, gridZ);
dim3 block(blockX, blockY, blockZ);
kernel<<<grid, block>>>(...)
```

**Grid 维度的限制（H100）**：

| 维度      | 最大值   | 总 Block 数限制                              |
| --------- | -------- | -------------------------------------------- |
| gridDim.x | 2^31 - 1 | gridDim.x × gridDim.y × gridDim.z ≤ 2^31 - 1 |
| gridDim.y | 65535    |                                              |
| gridDim.z | 65535    |                                              |

**实际示例：矩阵乘法的 Grid 配置**：

```cuda
// 计算 C (M×N) = A (M×K) × B (K×N)
#define TILE_SIZE 16

__global__ void matmul(float* A, float* B, float* C, int M, int N, int K) {
    // 每个block负责C的一个TILE_SIZE × TILE_SIZE子块
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// 启动配置
int M = 4096, N = 4096, K = 4096;
dim3 block(TILE_SIZE, TILE_SIZE);  // 16×16 = 256 threads per block
dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
          (M + TILE_SIZE - 1) / TILE_SIZE);  // 256×256 blocks

matmul<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

// 总共启动: 256×256 blocks × 256 threads/block = 16,777,216 threads
```

### Block：可调度的最小单位

**Block 是 GPU 调度的原子单位**，有以下关键特性：

1. **整块调度**：Block 内的所有线程必须在同一个 SM 上执行
2. **独立执行**：不同 Block 之间无法通信（除了全局内存）
3. **无序执行**：Block 的执行顺序不确定
4. **资源隔离**：每个 Block 有独立的 Shared Memory

**Block 的维度配置**：

```cuda
// 1D Block（适合1D数据，如向量）
dim3 block(256);  // 256个线程

// 2D Block（适合2D数据，如图像）
dim3 block(16, 16);  // 16×16 = 256个线程

// 3D Block（适合3D数据，如体积）
dim3 block(8, 8, 4);  // 8×8×4 = 256个线程
```

**Block 维度的限制（H100）**：

| 参数                    | 限制     | 说明                          |
| ----------------------- | -------- | ----------------------------- |
| blockDim.x              | 1024     | 单维度最大值                  |
| blockDim.y              | 1024     | 单维度最大值                  |
| blockDim.z              | 64       | Z 维度较小                    |
| **总线程数**            | **1024** | blockDim.x × y × z ≤ 1024     |
| **每 SM 最大 Block 数** | 32       | 受 Shared Memory 和寄存器限制 |

**为什么需要 Block 这一层？**

```
场景：1百万个数据元素，每个SM可以同时执行2048个线程

方案A：没有Block概念
  - 启动1,000,000个独立线程
  - 问题1: 线程间无法通信（如果需要协作）
  - 问题2: 调度粒度太细（调度开销大）
  - 问题3: 无法利用Shared Memory

方案B：使用Block（CUDA的设计）
  - 启动1000个Block，每个1000线程
  - 优势1: Block内线程可以通过Shared Memory协作
  - 优势2: 以Block为单位调度，开销低
  - 优势3: Block可以独立执行，易于扩展
```

**Block 内线程协作的示例**：

```cuda
__global__ void reduceSum(float* input, float* output, int N) {
    __shared__ float sdata[256];  // Shared Memory

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 每个线程加载一个元素到Shared Memory
    sdata[tid] = (i < N) ? input[i] : 0;
    __syncthreads();  // 等待Block内所有线程完成加载

    // 归约求和（树形归约）
    for (int s = blockDim.x / 2; s 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();  // 每轮归约后同步
    }

    // 线程0写出Block的结果
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}
```

这个例子展示了为什么 Block 是必需的：

- **Shared Memory**：Block 内线程共享数据
- **`__syncthreads()`**：Block 内的同步原语（不能跨 Block）
- **协作计算**：多个线程合作完成一个任务

### Thread：最小执行单元

每个线程有唯一的标识符，由 Block 和 Thread 的索引组成。

**线程索引计算**：

```cuda
// 1D索引
int tid = blockIdx.x * blockDim.x + threadIdx.x;

// 2D索引
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;

// 3D索引
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int z = blockIdx.z * blockDim.z + threadIdx.z;

// 转换为1D索引（用于访问数组）
int idx = z * (gridDim.x * blockDim.x) * (gridDim.y * blockDim.y)
        + y * (gridDim.x * blockDim.x)
        + x;
```

**内置变量总结**：

| 变量              | 类型  | 范围     | 说明                   |
| ----------------- | ----- | -------- | ---------------------- |
| `threadIdx.x/y/z` | uint3 | Block 内 | 线程在 Block 中的索引  |
| `blockIdx.x/y/z`  | uint3 | Grid 内  | Block 在 Grid 中的索引 |
| `blockDim.x/y/z`  | dim3  | 常量     | Block 的维度           |
| `gridDim.x/y/z`   | dim3  | 常量     | Grid 的维度            |
| `warpSize`        | int   | 常量     | Warp 大小（恒为 32）   |

**线程的资源分配**：

每个线程独享的资源：

| 资源                         | 每线程分配 | H100 限制                  |
| ---------------------------- | ---------- | -------------------------- |
| **寄存器**                   | 0-255 个   | 每 SM 总共 65,536 个寄存器 |
| **本地内存（Local Memory）** | 动态       | 实际上是全局内存的一部分   |
| **程序计数器（PC）**         | 1 个       | 硬件维护                   |
| **线程状态**                 | 少量 bit   | 活跃/等待/完成等           |

**寄存器分配示例**：

```cuda
__global__ void example() {
    int a, b, c;          // 3个寄存器
    float x, y, z;        // 3个寄存器
    double d;             // 2个寄存器（64-bit）
    // 总共使用约8个寄存器
}

// 如果kernel使用32个寄存器/线程
// SM最多可驻留：65536 / 32 = 2048个线程
// 等于64个warp（2048 / 32）
```

使用`nvcc --ptxas-options=-v`可以查看 kernel 的寄存器使用量：

```bash
$ nvcc -arch=sm_90 --ptxas-options=-v kernel.cu
ptxas info    : Used 32 registers, 1024 bytes smem,
                72 bytes cmem[0]
```

### 为什么是三层而不是两层或四层？

**两层模型的问题**（假设只有 Grid 和 Thread）：

```
Grid → Thread (无Block层)

问题1: 无法实现线程间协作
  - 没有Shared Memory的共享域
  - 无法使用__syncthreads()

问题2: 调度粒度过细
  - 需要为百万个线程单独调度
  - 硬件复杂度过高

问题3: 无法映射到SM
  - SM需要知道哪些线程组成一个工作单元
```

**四层模型的问题**（假设有 Grid → Cluster → Block → Thread）：

实际上，**Hopper 架构（H100）确实引入了第四层：Thread Block Cluster**！

```cuda
// Hopper的Cluster支持
__global__ void __cluster_dims__(2, 1, 1)  // 2个Block组成1个Cluster
clusterKernel() {
    // 可以访问Distributed Shared Memory
    // 跨Block通信
}
```

但 Cluster 是可选的高级特性，不是必须的层次。三层结构对大多数应用已经足够。

---

## 3. Warp：硬件执行的真实单位

### Warp 的本质

**关键洞察**：Thread 是软件抽象，Warp 是硬件现实

程序员写代码时考虑的是 Thread，但 GPU 硬件实际上以 Warp 为单位执行

**Warp 的定义**：

- **大小**：32 个连续的线程
- **划分规则**：Block 内的线程按 threadIdx.x 递增顺序，每 32 个分为一个 warp
- **执行方式**：warp 内所有线程执行相同指令（SIMT）
- **调度单位**：SM 的 Warp Scheduler 以 warp 为单位调度

**Warp 划分示例**：

```cuda
// Block配置：256个线程（1D）
dim3 block(256);

Warp划分：
Warp 0: Thread 0-31
Warp 1: Thread 32-63
Warp 2: Thread 64-95
...
Warp 7: Thread 224-255

总共8个warp
```

```cuda
// Block配置：16×16个线程（2D）
dim3 block(16, 16);

Warp划分（按row-major顺序）：
Warp 0: Thread (0,0) - (0,15), (1,0) - (1,15)
Warp 1: Thread (2,0) - (2,15), (3,0) - (3,15)
...

总共8个warp
```

**为什么是 32？**

这不是任意选择，而是硬件和软件的精心平衡：

| 考虑因素            | 太小（如 16）   | 32（实际） | 太大（如 64）   |
| ------------------- | --------------- | ---------- | --------------- |
| **SIMD 宽度利用率** | 低              | 适中       | 高              |
| **分支分化损失**    | 小              | 适中       | 大              |
| **寄存器压力**      | 小              | 适中       | 大              |
| **调度灵活性**      | 高              | 适中       | 低              |
| **硬件复杂度**      | 高（更多 warp） | 适中       | 低（更少 warp） |

历史上，NVIDIA 曾在不同架构中尝试其他大小，但自 G80（2006）以来，32 一直是最优选择。

### Warp 调度的微观机制

**SM 的 Warp 调度器架构**（以 H100 为例）：

```
SM内部结构（简化）
┌─────────────────────────────────────────┐
│  Warp Scheduler 0  Warp Scheduler 1     │
│  Warp Scheduler 2  Warp Scheduler 3     │
├─────────────────────────────────────────┤
│  Warp池（最多64个warp）                  │
│  ┌─────┬─────┬─────┬─────┬─────┐       │
│  │Warp │Warp │Warp │Warp │ ... │       │
│  │  0  │  1  │  2  │  3  │     │       │
│  │Ready│Wait │Ready│Wait │     │       │
│  └─────┴─────┴─────┴─────┴─────┘       │
├─────────────────────────────────────────┤
│  执行单元                                │
│  ┌─────────────┐  ┌──────────────┐     │
│  │ CUDA Cores  │  │ Tensor Cores │     │
│  │ (128个)     │  │ (4个)         │     │
│  └─────────────┘  └──────────────┘     │
└─────────────────────────────────────────┘
```

**Warp 的生命周期**：

```
1. 创建阶段（Block分配到SM时）
   ├─ 分配寄存器
   ├─ 分配Shared Memory
   ├─ 初始化PC和状态
   └─ 加入Warp池

2. Ready状态（可执行）
   ├─ 所有操作数已就绪
   ├─ 需要的执行单元空闲
   └─ 无同步障碍

3. Stall状态（等待中）
   ├─ 内存访问未完成
   ├─ 等待__syncthreads()
   ├─ 数据依赖未解决
   └─ 执行单元忙碌

4. 执行阶段
   ├─ Warp Scheduler选中
   ├─ 发射指令到执行单元
   └─ 1-4个cycle后进入下一状态

5. 完成阶段（所有线程执行完毕）
   ├─ 释放寄存器
   └─ 从Warp池移除
```

**Warp 调度算法**：

H100 的调度器使用**优先级调度**结合**Round Robin**：

```python
# 伪代码：Warp调度算法
def select_warp_to_execute():
  ready_warps = []

  # 第一步：筛选Ready状态的warp
  for warp in warp_pool:
    if warp.state == READY:
      if check_resources_available(warp):
        ready_warps.append(warp)

    if len(ready_warps) == 0:
      return None  # 所有warp都在等待

  # 第二步：按优先级排序
  # 优先级因素：
  # 1. 指令类型（内存指令 < 计算指令）
  # 2. Warp年龄（等待时间长的优先）
  # 3. 公平性（轮询计数器）

  ready_warps.sort(key=lambda w: (
    w.instruction_priority,
    -w.wait_cycles,
    w.round_robin_counter
  ))

  # 第三步：选择最高优先级的warp
  selected = ready_warps[0]
  selected.round_robin_counter += 1

  return selected
```

**实际调度示例**：

假设 SM 上有 4 个 warp：

```
Cycle 0:
Warp 0: READY  (执行FMA指令)
Warp 1: READY  (执行FMA指令)
Warp 2: STALL  (等待内存加载)
Warp 3: READY  (执行ADD指令)

Scheduler选择: Warp 0 (轮询，最先Ready)
执行: FMA指令（4 cycle延迟）

Cycle 1:
Warp 0: STALL  (FMA流水线，还需3 cycles)
Warp 1: READY
Warp 2: STALL
Warp 3: READY

Scheduler选择: Warp 1
执行: FMA指令

Cycle 2:
Warp 0: STALL  (还需2 cycles)
Warp 1: STALL  (还需3 cycles)
Warp 2: STALL  (内存未返回)
Warp 3: READY

Scheduler选择: Warp 3
执行: ADD指令（1 cycle延迟）

Cycle 3:
Warp 0: STALL  (还需1 cycle)
Warp 1: STALL  (还需2 cycles)
Warp 2: READY  (内存数据返回！)
Warp 3: READY

Scheduler选择: Warp 2
执行: MUL指令

Cycle 4:
Warp 0: READY  (FMA完成)
Warp 1: STALL
Warp 2: READY
Warp 3: READY

Scheduler选择: Warp 0 (轮询回到Warp 0)
```

**关键指标：Warp Execution Efficiency**

```
Warp执行效率 = 实际执行cycles / 理论最小cycles

理想情况（无stall）:
  100% 效率 - 每个cycle都有warp在执行

实际情况:
  60-80% - 良好优化的kernel
  30-50% - 内存密集型kernel
  < 30%  - 存在严重瓶颈
```

可以用`nvprof`或`Nsight Compute`测量：

```bash
$ ncu --metrics sm__warps_active.avg.pct_of_peak kernel.exe
  sm__warps_active.avg.pct_of_peak: 45.2%
  # 表示SM上平均有45.2%的时间有warp在执行
```

### 线程分歧（Divergence）：性能杀手

**分歧的定义**：

同一 warp 内的线程执行不同的控制流路径。

**分歧的代价示例**：

```cuda
__global__ void divergent_kernel(int* data, int threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int value = data[idx];

    if (value threshold) {
        // 路径A：复杂计算
        for (int i = 0; i < 100; i++) {
            value = value * value + i;
        }
    } else {
        // 路径B：简单计算
        value = value + 1;
    }

    data[idx] = value;
}
```

**硬件执行过程**：

假设 warp 中 20 个线程走路径 A，12 个线程走路径 B：

```
Step 1: 评估条件 (value threshold)
  Cycle 0: 所有32个线程执行比较
  Active Mask: 11111111111111111111111111111111

Step 2: 执行路径A (value threshold为true)
  Cycle 1-100: 只有20个线程活跃
  Active Mask: 11111111111111111111000000000000
  其余12个线程IDLE（空闲但不能执行其他任务）

Step 3: 执行路径B (value threshold为false)
  Cycle 101: 只有12个线程活跃
  Active Mask: 00000000000000000000111111111111
  前20个线程IDLE

Step 4: 合并路径（继续执行后续代码）
  Cycle 102: 所有32个线程重新活跃
  Active Mask: 11111111111111111111111111111111
```

**性能分析**：

| 场景                           | 执行时间   | 有效利用率                       |
| ------------------------------ | ---------- | -------------------------------- |
| **无分歧（所有线程走路径 A）** | 100 cycles | 100%                             |
| **无分歧（所有线程走路径 B）** | 1 cycle    | 100%                             |
| **分歧（20A + 12B）**          | 101 cycles | (20×100 + 12×1)/(32×101) = 62.4% |
| **最坏分歧（1A + 31B）**       | 101 cycles | (1×100 + 31×1)/(32×101) = 4.1%   |

**分歧的类型**：

1. **数据相关分歧**：

```cuda
if (threadIdx.x < 16) {  // 完全可预测的分歧
    // 前16个线程执行
} else {
    // 后16个线程执行
}
```

这种分歧编译器可以优化为两个独立的代码块。

2. **数据依赖分歧**：

```cuda
if (data[idx] threshold) {  // 运行时才知道的分歧
    // 无法预测哪些线程会执行
}
```

这是最常见也最难优化的分歧。

3. **循环分歧**：

```cuda
for (int i = 0; i < data[idx]; i++) {  // 不同线程不同迭代次数
    // 最慢的线程决定整个warp的时间
}
```

### 如何避免或减少分歧

**策略 1：重组数据以减少分歧**

```cuda
// 差的做法：交替的数据
int data[64] = {1, 100, 2, 99, 3, 98, ...};  // 奇偶不同

// Warp 0处理前32个元素
// 分歧严重：16个线程走路径A，16个走路径B

// 好的做法：聚集相似数据
int data[64] = {1, 2, 3, ..., 32, 33, ..., 100};  // 先小后大

// 前16个Warp处理小值（都走路径B）
// 后16个Warp处理大值（都走路径A）
// 无分歧！
```

**策略 2：使用 Warp-Level 原语**

```cuda
// 差的做法
if (threadIdx.x == 0) {
    sum = 0;
    for (int i = 0; i < 32; i++) {
        sum += value[i];
    }
}

// 好的做法：使用Warp Shuffle
float sum = value;
for (int offset = 16; offset 0; offset /= 2) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
}
// 所有线程都参与，无分歧
```

**策略 3：分支提示（Branch Hints）**

```cuda
if (__builtin_expect(condition, 1)) {  // 提示:大概率为true
    // 路径A
} else {
    // 路径B（很少执行）
}
```

虽然不能消除分歧，但可以帮助编译器优化指令布局。

**策略 4：计算替代分支**

```cuda
// 差的做法
if (x 0) {
    y = sqrt(x);
} else {
    y = 0;
}

// 好的做法：计算两个分支，用掩码选择
float result_a = sqrt(x);
float result_b = 0;
y = (x 0) ? result_a : result_b;  // GPU可能将其编译为SELP指令
```

注意：只有当计算成本低于分歧成本时才有效。

**性能对比**（实际测试，H100）：

```
Kernel: 1024个Block × 256个Thread = 262,144个Thread
数据: 50%的线程走路径A（1000次迭代），50%走路径B（1次迭代）

结果:
  分歧版本: 15.2 ms
  重组数据后: 7.8 ms  (提升1.95×)
  使用Warp Shuffle: 0.3 ms  (提升50.6×)
```

---

## 4. SIMT vs SIMD：GPU 的独特执行模型

### SIMD 的局限性

在深入 CUDA 执行模型之前，我们需要理解 GPU 为什么不简单采用 SIMD（Single Instruction Multiple Data）。

**SIMD 的经典例子（Intel AVX-512）**：

```c
// CPU SIMD代码：一次处理16个float
__m512 a = _mm512_load_ps(&A[i]);     // 加载16个元素
__m512 b = _mm512_load_ps(&B[i]);     // 加载16个元素
__m512 c = _mm512_mul_ps(a, b);       // 16个乘法
_mm512_store_ps(&C[i], c);            // 存储16个结果
```

这段代码在单个 CPU 核心上同时处理 16 个数据。但 SIMD 有严格的限制：

**SIMD 的限制**：

| 限制             | 说明                                 | 影响                 |
| ---------------- | ------------------------------------ | -------------------- |
| **固定向量长度** | AVX-512 固定为 512-bit (16 个 float) | 无法适应不同数据规模 |
| **同步执行**     | 所有通道必须执行相同指令             | 无法处理条件分支     |
| **显式编程**     | 程序员需手动向量化                   | 编程复杂度高         |
| **内存对齐要求** | 数据必须 64 字节对齐                 | 灵活性差             |

**分支处理的灾难**：

```c
// 有条件分支的代码
for (int i = 0; i < N; i++) {
  if (x[i] 0) {
    y[i] = sqrt(x[i]);    // 路径A
  } else {
    y[i] = 0;             // 路径B
  }
}
```

在 SIMD 中，这段代码极难高效实现：

```c
// SIMD版本（效率低下）
__m512 x_vec = _mm512_load_ps(&x[i]);
__mmask16 mask = _mm512_cmp_ps_mask(x_vec, zero, _CMP_GT_OQ);

// 必须同时计算两个分支
__m512 result_a = _mm512_sqrt_ps(x_vec);  // 全部计算sqrt
__m512 result_b = zero;                    // 全部设为0

// 根据mask选择结果
__m512 y_vec = _mm512_mask_blend_ps(mask, result_b, result_a);
```

即使只有 1 个元素满足条件，所有 16 个通道都要执行`sqrt`，浪费了 15 个通道的计算。

### SIMT：更灵活的并行模型

**术语：SIMT (Single Instruction Multiple Threads)**
**定义**：多个线程执行相同指令，但每个线程有独立的程序计数器和寄存器
**关键特性**：线程可以独立执行不同路径，但同一 warp 内的线程同步执行效率最高
**发明者**：NVIDIA，首次在 Tesla 架构（2006）中引入

**SIMT vs SIMD 对比**：

```
SIMD执行（Intel AVX-512）:
┌────────────────────────────────────────┐
│  单个指令 → 16个数据通道                │
│                                        │
│  MUL  →  [D0] [D1] [D2] ... [D15]     │
│                                        │
│  所有通道必须执行相同操作               │
└────────────────────────────────────────┘

SIMT执行（NVIDIA CUDA）:
┌────────────────────────────────────────┐
│  单个指令 → 32个线程                    │
│                                        │
│  MUL  →  [T0] [T1] [T2] ... [T31]     │
│           ↓    ↓    ↓         ↓       │
│         [PC0][PC1][PC2]...[PC31]      │
│         [R0] [R1] [R2] ... [R31]      │
│                                        │
│  每个线程有独立的PC和寄存器              │
│  可以执行不同路径（有性能代价）          │
└────────────────────────────────────────┘
```

**SIMT 处理分支的示例**：

```cuda
// CUDA代码
__global__ void kernel(float* x, float* y, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        if (x[i] 0) {
            y[i] = sqrtf(x[i]);  // 路径A
        } else {
            y[i] = 0;             // 路径B
        }
    }
}
```

**硬件执行（warp 级别）**：

假设 warp 中 32 个线程，其中 20 个走路径 A，12 个走路径 B：

```
Cycle 0-5: 执行路径A（20个线程活跃）
  Active Mask: 11111111111111111111000000000000
  执行: y[i] = sqrtf(x[i])

Cycle 6-7: 执行路径B（12个线程活跃）
  Active Mask: 00000000000000000000111111111111
  执行: y[i] = 0

总耗时: 5 + 2 = 7 cycles
如果没有分支（32个线程都走路径A）: 5 cycles
分支效率: 5/7 = 71.4%
```

**SIMT 的优势**：

1. **程序员友好**：可以像写串行代码一样写并行代码
2. **硬件灵活性**：自动处理分支，无需手动向量化
3. **可扩展性**：从 32 个线程到数百万线程，代码无需修改
4. **动态调度**：硬件自动选择执行哪个 warp

**代价**：

1. **分支分化开销**：不同路径需要串行执行
2. **硬件复杂度**：需要为每个线程维护状态
3. **寄存器压力**：每个线程需要独立寄存器

---

## 5. 占用率（Occupancy）：资源利用的艺术

### Occupancy 的定义

**术语：Occupancy（占用率）**
**定义**：活跃 warp 数量与 SM 最大支持 warp 数量的比值
**公式**：Occupancy = 活跃 warp 数 / SM 最大 warp 数（H100 为 64）
**范围**：0% - 100%

**为什么 Occupancy 重要？**

回顾延迟隐藏机制：

```
内存访问延迟: 400 cycles
单条指令执行时间: 4 cycles
需要隐藏延迟的warp数: 400 / 4 = 100

但H100每个SM最多只能驻留64个warp！

解决方案:
- 如果Occupancy = 100% (64个warp)
  实际可隐藏: 64 × 4 = 256 cycles (64%的延迟)
- 如果Occupancy = 50% (32个warp)
  实际可隐藏: 32 × 4 = 128 cycles (32%的延迟)
```

低 Occupancy 意味着调度器没有足够的 warp 来填补空闲时间，导致 SM 利用率降低。

### 影响 Occupancy 的因素

**H100 的资源限制**：

| 资源类型          | 每 SM 总量 | 每 Block 限制 | 每 Thread 限制 |
| ----------------- | ---------- | ------------- | -------------- |
| **寄存器**        | 65,536 个  | 无            | 255 个         |
| **Shared Memory** | 0-228 KB   | 228 KB        | 无             |
| **线程**          | 2048 个    | 1024 个       | -              |
| **Block**         | 32 个      | -             | -              |
| **Warp**          | 64 个      | 32 个         | -              |

**Occupancy 计算公式**：

```
活跃Warp数 = min(
    ⌊65536 / (寄存器/thread × 32)⌋,  # 寄存器限制
    ⌊Shared_Mem_Size / Shared_Mem_per_block⌋ × warps_per_block,  # Shared Memory限制
    ⌊2048 / threads_per_block⌋ × warps_per_block,  # 线程数限制
    32 × warps_per_block  # Block数限制
)

Occupancy = 活跃Warp数 / 64
```

**实际计算示例**：

```cuda
// Kernel配置
__global__ void kernel() {
    __shared__ float sdata[2048];  // 8 KB Shared Memory

    float a, b, c, d;  // 假设使用40个寄存器/线程
    // ...
}

// 启动配置
dim3 block(256);  // 256 threads/block = 8 warps/block

计算Occupancy:

1. 寄存器限制:
   每个warp需要: 40 reg/thread × 32 thread/warp = 1280 registers
   最多warp数: 65536 / 1280 = 51.2 → 51 warps

2. Shared Memory限制:
   每个block需要: 8 KB
   最多block数: 228 KB / 8 KB = 28.5 → 28 blocks
   最多warp数: 28 blocks × 8 warps/block = 224 warps (超过64，无限制)

3. 线程数限制:
   每个block: 256 threads
   最多block数: 2048 / 256 = 8 blocks
   最多warp数: 8 blocks × 8 warps/block = 64 warps

4. Block数限制:
   最多32个block，每个8 warp
   最多warp数: 32 × 8 = 256 warps (超过64，无限制)

瓶颈: 寄存器（51 warps）

实际Occupancy = 51 / 64 = 79.7%
```

### CUDA Occupancy Calculator 实战

NVIDIA 提供了多种工具计算 Occupancy：

**方法 1：Excel 表格（传统方法）**

下载`CUDA_Occupancy_Calculator.xls`，输入：

- Compute Capability: 9.0 (H100)
- Threads per block: 256
- Registers per thread: 40
- Shared Memory per block: 8192 bytes

输出：

- Occupancy: 79.69%
- Limiting factor: Registers

**方法 2：nvcc 编译时查看**

```bash
$ nvcc -arch=sm_90 --ptxas-options=-v kernel.cu

ptxas info : Used 40 registers, 8192 bytes smem
ptxas info : Function properties for kernel:
    0 bytes stack frame, 0 bytes spill stores,
    0 bytes spill loads
```

**方法 3：运行时 API**

```cuda
int device = 0;
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, device);

int blockSize = 256;
int numBlocks;

// 自动计算最优配置
cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &numBlocks,
    kernel,
    blockSize,
    dynamicSharedMemSize
);

float occupancy = (numBlocks * blockSize / prop.warpSize) /
                  (float)prop.maxThreadsPerMultiProcessor;
printf("Occupancy: %.2f%%\n", occupancy * 100);
```

**方法 4：Nsight Compute 分析**

```bash
$ ncu --metrics sm__warps_active.avg.pct_of_peak kernel.exe

sm__warps_active.avg.pct_of_peak: 79.7%
Limiter: Register Usage
```

### Occupancy 优化策略

**策略 1：减少寄存器使用**

```cuda
// 差的做法：大量中间变量
__global__ void kernel() {
    float a1, a2, a3, ..., a50;  // 50个局部变量
    // 使用40+个寄存器
}

// 好的做法：重用变量
__global__ void kernel() {
    float temp;  // 只用1-2个临时变量
    temp = ...;
    result += temp;
    temp = ...;  // 重用
    result += temp;
}

// 或者使用编译器指令
__global__ void __launch_bounds__(256, 8)  // 256 threads/block, 最少8个block/SM
kernel() {
    // 编译器会限制寄存器使用以达到目标occupancy
}
```

**策略 2：调整 Block 大小**

```cuda
// 测试不同Block大小的Occupancy

Block size = 128:
  寄存器限制: 64 warps
  线程数限制: 2048/128 = 16 blocks × 4 warps = 64 warps
  Occupancy: 100%

Block size = 256:
  寄存器限制: 51 warps
  线程数限制: 2048/256 = 8 blocks × 8 warps = 64 warps
  Occupancy: 79.7%

Block size = 512:
  寄存器限制: 51 warps
  线程数限制: 2048/512 = 4 blocks × 16 warps = 64 warps
  Occupancy: 79.7%

结论: Block=128时Occupancy最高
```

但注意：**高 Occupancy != 高性能**！

**策略 3：减少 Shared Memory 使用**

```cuda
// 差的做法：分配过大的Shared Memory
__shared__ float sdata[1024];  // 4 KB
// 限制: 228 KB / 4 KB = 57 blocks

// 好的做法：精确计算需要的大小
__shared__ float sdata[256];  // 1 KB
// 限制: 228 KB / 1 KB = 228 blocks (无限制)
```

或使用动态 Shared Memory：

```cuda
__global__ void kernel() {
    extern __shared__ float sdata[];  // 动态大小
}

// 启动时指定大小
kernel<<<grid, block, sharedMemSize>>>();
```

### Occupancy vs 性能的真实关系

**反直觉的事实**：高 Occupancy 并不总是带来高性能！

**实验：矩阵乘法的 Occupancy vs 性能**

```cuda
__global__ void matmul(float* A, float* B, float* C, int N) {
    // 三个版本：不同的Shared Memory使用
}

版本1: 无Shared Memory
  Occupancy: 100%
  Performance: 5 TFLOPS
  瓶颈: 全局内存带宽

版本2: 8 KB Shared Memory/block
  Occupancy: 79.7%
  Performance: 15 TFLOPS (3×提升!)
  瓶颈: 计算

版本3: 32 KB Shared Memory/block
  Occupancy: 50%
  Performance: 20 TFLOPS (4×提升!)
  瓶颈: 计算
```

**为什么低 Occupancy 性能更高？**

1. **减少全局内存访问**：Shared Memory 的数据重用大幅降低了带宽需求
2. **更多寄存器可用**：每个线程可以使用更多寄存器，减少内存访问
3. **更好的缓存命中率**：更少的活跃 warp 意味着每个 warp 的缓存份额更大

**Occupancy 的经验法则**：

| Occupancy 范围 | 适用场景          | 优化建议                     |
| -------------- | ----------------- | ---------------------------- |
| **< 25%**      | 极可能存在问题    | 检查资源使用，可能过度优化了 |
| **25% - 50%**  | 计算密集型 kernel | 可以接受，关注计算效率       |
| **50% - 75%**  | 平衡型 kernel     | 良好的起点                   |
| **75% - 100%** | 内存密集型 kernel | 需要高 Occupancy 隐藏延迟    |

**正确的优化流程**：

```
1. 先实现功能正确的kernel
2. 使用Profiler识别瓶颈
3. 如果瓶颈是内存延迟 → 提升Occupancy
4. 如果瓶颈是带宽 → 优化访问模式（可能降低Occupancy）
5. 如果瓶颈是计算 → 优化算法（Occupancy可能不重要）
```

---

## 6. 实战案例：从理论到实践

### 案例 1：向量加法

**最简单的 CUDA 程序**：

```cuda
__global__ void vectorAdd(float* A, float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

// 启动配置
int N = 1024 * 1024;  // 1M elements
int threadsPerBlock = 256;
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
```

**执行分析**：

```
配置:
- Grid: 4096 blocks
- Block: 256 threads
- 总线程: 1,048,576

Warp划分:
- 每个block: 256 / 32 = 8 warps
- 总warp数: 4096 × 8 = 32,768 warps

资源需求:
- 寄存器: ~4 per thread
- Shared Memory: 0
- Occupancy: 100% (资源充足)

性能:
- 计算: 1M FLOPs (加法)
- 内存: 3M × 4 bytes = 12 MB (读A, B, 写C)
- Arithmetic Intensity: 1M / 12M = 0.083 FLOP/Byte
- 瓶颈: 内存带宽
```

在 H100 上（3 TB/s 带宽）：

```
理论时间 = 12 MB / 3 TB/s = 4 μs
实际时间 ≈ 6 μs (考虑kernel启动开销)
```

### 案例 2：矩阵乘法的演进

**版本 1：朴素实现**

```cuda
__global__ void matmul_naive(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// 配置
dim3 block(16, 16);  // 256 threads
dim3 grid((N + 15) / 16, (N + 15) / 16);
```

**性能分析**：

```
N = 4096

计算量:
- 每个元素: 4096次FMA = 8192 FLOPs
- 总计: 4096^2 × 8192 = 137 GFLOPs

内存访问:
- 读A: 4096^2 × 4096 × 4 bytes = 256 GB
- 读B: 4096^2 × 4096 × 4 bytes = 256 GB
- 写C: 4096^2 × 4 bytes = 64 MB
- 总计: ≈ 512 GB

Arithmetic Intensity: 137G / 512G = 0.27 FLOP/Byte
瓶颈: 严重的内存带宽瓶颈

H100上的性能:
- 理论带宽需求: 512 GB
- 实际时间: 512 GB / 3 TB/s ≈ 170 ms
- 实测时间: ≈ 200 ms
- 算力利用率: 137 GFLOPS / 378 TFLOPS = 0.036%

问题: 数据重用率极低，每个数据从内存加载一次就丢弃
```

**版本 2：使用 Shared Memory 分块**

```cuda
#define TILE_SIZE 16

__global__ void matmul_tiled(float* A, float* B, float* C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // 分块遍历
    for (int t = 0; t < N / TILE_SIZE; t++) {
        // 协作加载tile到Shared Memory
        As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        __syncthreads();

        // 使用Shared Memory计算
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    C[row * N + col] = sum;
}
```

**性能改进**：

```
数据重用分析:
- 每个tile加载一次到Shared Memory
- 被tile内256个线程重用
- 重用次数: 16次 (TILE_SIZE)

内存访问量:
- 读A: 4096^2 × 4096 / 16 × 4 = 16 GB (减少16×)
- 读B: 4096^2 × 4096 / 16 × 4 = 16 GB (减少16×)
- 写C: 64 MB
- 总计: ≈ 32 GB

Arithmetic Intensity: 137G / 32G = 4.3 FLOP/Byte (提升16×)

H100上的性能:
- 实测时间: ≈ 12 ms (提升16.7×)
- 算力利用率: 137 GFLOPS / 12 ms = 11.4 TFLOPS
- 利用率: 11.4 / 378 = 3%

仍有改进空间!
```

**版本 3：避免 Bank Conflict + 更大 Tile**

```cuda
#define TILE_SIZE 32

__global__ void matmul_optimized(float* A, float* B, float* C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];  // +1避免bank conflict
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < N / TILE_SIZE; t++) {
        // 使用向量化加载（128-bit）
        float4 a_val = *((float4*)&A[row * N + t * TILE_SIZE + (threadIdx.x & ~3)]);
        As[threadIdx.y][threadIdx.x] = (threadIdx.x & 3) == 0 ? a_val.x :
                                        (threadIdx.x & 3) == 1 ? a_val.y :
                                        (threadIdx.x & 3) == 2 ? a_val.z : a_val.w;

        float4 b_val = *((float4*)&B[(t * TILE_SIZE + threadIdx.y) * N + col]);
        Bs[threadIdx.y][threadIdx.x] = /* similar */;
        __syncthreads();

        // 展开内循环
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    C[row * N + col] = sum;
}
```

**最终性能**：

```
改进:
1. TILE_SIZE 32: 重用次数翻倍，内存减半
2. +1 padding: 消除bank conflict，Shared Memory带宽提升2×
3. 向量化加载: 合并访问，全局内存带宽提升4×
4. 循环展开: 减少循环开销

结果:
- 实测时间: ≈ 0.4 ms (总提升500×)
- 性能: 137 GFLOPS / 0.4 ms = 342 TFLOPS
- 算力利用率: 342 / 378 = 90.5%

接近cuBLAS性能 (350 TFLOPS)!
```

### 案例 3：Reduction 优化

**问题**：对 N 个元素求和

**版本 1：使用 Shared Memory**

```cuda
__global__ void reduce_v1(float* input, float* output, int N) {
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 加载数据到Shared Memory
    sdata[tid] = (i < N) ? input[i] : 0;
    __syncthreads();

    // 树形归约
    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}
```

**问题分析**：

```
分支分化:
- 第1轮: tid % 2 == 0, 一半线程活跃
- 第2轮: tid % 4 == 0, 1/4线程活跃
- ...
- 第8轮: tid % 256 == 0, 只有1个线程活跃

Warp效率:
- Warp 0: 第1轮32个线程活跃，第2轮16个，...
- 严重的线程分化

性能: ~50% 理论峰值
```

**版本 2：消除分支分化**

```cuda
__global__ void reduce_v2(float* input, float* output, int N) {
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < N) ? input[i] : 0;
    __syncthreads();

    // 改进的归约（相邻线程协作）
    for (int s = blockDim.x / 2; s 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}
```

**改进**：

```
现在:
- 第1轮: tid < 128, 前4个warp活跃（无分化）
- 第2轮: tid < 64, 前2个warp活跃
- 第3轮: tid < 32, 前1个warp活跃（完美）
- 第4轮: tid < 16, warp内分化开始

Warp效率: ~85%
性能提升: 1.7×
```

**版本 3：使用 Warp-Level 原语**

```cuda
__device__ float warpReduce(float val) {
    for (int offset = 16; offset 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void reduce_v3(float* input, float* output, int N) {
    __shared__ float sdata[8];  // 只需8个元素（每个warp一个）

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float val = (i < N) ? input[i] : 0;

    // Warp内归约（无同步开销）
    val = warpReduce(val);

    // 每个warp的第一个线程写入Shared Memory
    if (tid % 32 == 0) {
        sdata[tid / 32] = val;
    }
    __syncthreads();

    // 第一个warp归约Shared Memory
    if (tid < 8) {
        val = sdata[tid];
        val = warpReduce(val);
        if (tid == 0) {
            output[blockIdx.x] = val;
        }
    }
}
```

**最终性能**：

```
优势:
1. 无Shared Memory bank conflict
2. 无分支分化
3. 无__syncthreads()开销（warp内同步是隐式的）

性能: ~98% 理论峰值
相比v1提升: 2.5×
```

### 性能优化总结

| 优化技术               | 加速比   | 适用场景                |
| ---------------------- | -------- | ----------------------- |
| **使用 Shared Memory** | 2-20×    | 数据重用高的算法        |
| **消除 Bank Conflict** | 1.5-3×   | 大量 Shared Memory 访问 |
| **避免线程分歧**       | 1.5-10×  | 有条件分支的 kernel     |
| **合并内存访问**       | 3-10×    | 随机访问模式            |
| **使用 Warp Shuffle**  | 1.2-2×   | 线程间通信              |
| **循环展开**           | 1.1-1.3× | 小循环体                |
| **Tensor Core**        | 10-30×   | 矩阵运算                |
| **提升 Occupancy**     | 1.5-3×   | 内存延迟受限            |

---

## 7. 总结与下一步

### 核心概念回顾

**CUDA 执行模型的三个关键抽象**：

1. **Grid-Block-Thread 层次结构**：

   - Grid：整个 kernel 的执行空间
   - Block：可调度的最小单位，可以访问 Shared Memory
   - Thread：最小执行单元，有独立的寄存器

2. **Warp：硬件执行的真实单位**：

   - 32 个线程打包成 1 个 warp
   - SIMT 执行：同时发射相同指令
   - 分支分化会降低效率

3. **Occupancy：资源利用率**：
   - 受寄存器、Shared Memory、线程数限制
   - 高 Occupancy ≠ 高性能
   - 需要根据瓶颈类型优化

### 设计哲学

**CPU vs GPU 的根本差异**：

| 维度         | CPU         | GPU                |
| ------------ | ----------- | ------------------ |
| **线程开销** | 8 MB/thread | 1 KB/thread        |
| **切换开销** | 微秒级      | 零开销（硬件调度） |
| **并行度**   | 数十        | 数十万             |
| **编程模型** | 独立线程    | 数据并行（SIMT）   |
| **优化目标** | 单线程延迟  | 整体吞吐量         |

**为什么 SIMT 优于 SIMD**：

- 编程简单：像写串行代码
- 硬件灵活：自动处理分支（尽管有代价）
- 可扩展：从 32 线程到数百万线程

### 实践建议

**选择 Block 大小**：

```
经验法则:
- 从128或256开始
- 确保是warp大小（32）的倍数
- 2D/3D问题：保持维度是2的幂（如16×16）
- 使用Occupancy Calculator验证
```

**避免常见陷阱**：

1. **分支分化**：重组数据或使用 warp-level 原语
2. **低 Occupancy**：减少寄存器/Shared Memory 使用
3. **Bank Conflict**：使用+1 padding 或重组访问模式
4. **非合并访问**：确保连续线程访问连续内存

**性能分析流程**：

```
1. 使用nvprof/Nsight Compute识别瓶颈
2. 如果是计算受限：
   → 使用Tensor Core
   → 优化算法
3. 如果是内存带宽受限：
   → 使用Shared Memory增加数据重用
   → 确保合并访问
4. 如果是内存延迟受限：
   → 提升Occupancy
   → 使用异步拷贝
```

### 扩展阅读

**NVIDIA 官方文档**：

- CUDA C Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- CUDA Best Practices Guide: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/

**性能分析工具**：

- Nsight Compute: https://developer.nvidia.com/nsight-compute
- CUDA Occupancy Calculator: 包含在 CUDA Toolkit 中

**学术论文**：

- "Demystifying GPU Microarchitecture through Microbenchmarking" (ISPASS 2010)
- "Understanding the GPU Microarchitecture to Achieve Bare-Metal Performance Tuning" (PPoPP 2017)

---
