---
title: CUDA执行模型：理解Warp、SMIT与Occupancy
date: 2025-10-21 17:01:13
tags:
---

GPU 硬件从不执行单个 Thread。它执行的是 Warp —— 32 个 Thread 打包成的执行单元。

理解 Warp，你就能解释那些反直觉的性能现象。

<!--more-->

---

## 1. Warp：GPU 执行的原子单位

### Thread 是抽象，Warp 是真实

**关键认知转变**：

```
你写代码时思考的单位：        GPU硬件执行的单位：
          Thread                Warp (32个Thread)
            ↓                      ↓
          独立计算                打包执行
```

当你写：

```cuda
__global__ void kernel() {
    int i = threadIdx.x;
    // 每个Thread独立计算
}

kernel<<<1, 256>>>();
```

你认为启动了 256 个独立的 Thread。

但 GPU 硬件看到的是：

```
256个Thread = 8个Warp
Warp 0: Thread 0-31
Warp 1: Thread 32-63
Warp 2: Thread 64-95
...
Warp 7: Thread 224-255
```

### Warp 的精确定义

**Warp 的划分规则**：

Block 内的 Thread 按`threadIdx.x`递增顺序，每 32 个打包成一个 Warp。

**1D Block 的划分**（256 个 Thread）：

```
Block内Thread编号：
0  1  2  ... 30 31 | 32 33 34 ... 62 63 | 64 65 ... 95 | ... | 224 ... 255
└─────Warp 0──────┘ └─────Warp 1──────┘ └───Warp 2───┘       └──Warp 7──┘

总共：256 / 32 = 8个Warp
```

**2D Block 的划分**（16×16=256 个 Thread）：

```
Block内Thread布局（threadIdx.x, threadIdx.y）：

      x: 0    1    2   ...  14   15  |  0    1    2   ...  14   15
y=0: (0,0)(1,0)(2,0)...(14,0)(15,0) |(0,1)(1,1)(2,1)...(14,1)(15,1) | ...
     └──────────Warp 0──────────────┘ └──────────Warp 1──────────────┘

按row-major顺序（先x后y）：
Warp 0: (0,0)-(15,0), (0,1)-(15,1)  ← 前两行的32个Thread
Warp 1: (0,2)-(15,2), (0,3)-(15,3)  ← 第3-4行的32个Thread
...
```

**关键规则**：

- Warp 的划分完全由硬件决定，程序员无法控制
- 划分顺序：先 x，再 y，最后 z
- 一个 Warp 内的 Thread 在物理上相邻

### 为什么是 32？

这不是任意选择。32 是多个因素权衡的结果：

| 考虑因素          | Warp=16                | Warp=32 | Warp=64                |
| ----------------- | ---------------------- | ------- | ---------------------- |
| **SIMD 宽度利用** | 低                     | 中      | 高                     |
| **分支分化损失**  | 小                     | 中      | 大                     |
| **寄存器压力**    | 小（每个 Warp 占用少） | 中      | 大（每个 Warp 占用多） |
| **调度灵活性**    | 高（更多 Warp 可选）   | 中      | 低（Warp 数少）        |
| **硬件复杂度**    | 高（需要更多调度器）   | 中      | 低                     |

**历史数据**：

- Tesla 架构（2006）：确定为 32
- 至今所有 NVIDIA GPU：保持 32 不变
- AMD GPU：64（不同的设计哲学）

这个数字深刻影响了 CUDA 编程的方方面面。

---

## 2. SIMT 执行模型：Warp 如何执行

**术语**：SIMT (Single Instruction, Multiple Threads)

- **Single Instruction**：一条指令
- **Multiple Threads**：多个线程
- **核心**：Warp 内 32 个 Thread 同时执行相同的指令

### Warp 执行的微观过程

假设有这样一个 kernel：

```cuda
__global__ void add(float* A, float* B, float* C) {
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}
```

**Warp 0（Thread 0-31）的执行**：

```
Cycle 0: 所有32个Thread执行：int i = threadIdx.x
  Thread 0: i = 0
  Thread 1: i = 1
  ...
  Thread 31: i = 31

Cycle 1-4: 所有32个Thread执行：读取A[i]
  Thread 0读A[0], Thread 1读A[1], ..., Thread 31读A[31]
  （4个cycle因为内存延迟）

Cycle 5-8: 所有32个Thread执行：读取B[i]
  Thread 0读B[0], Thread 1读B[1], ..., Thread 31读B[31]

Cycle 9: 所有32个Thread执行：加法
  Thread 0: C[0] = A[0] + B[0]
  Thread 1: C[1] = A[1] + B[1]
  ...
  Thread 31: C[31] = A[31] + B[31]

Cycle 10-13: 所有32个Thread执行：写C[i]
```

**关键观察**：

1. **相同指令**：每个 cycle，32 个 Thread 执行完全相同的指令
2. **不同数据**：但每个 Thread 操作不同的数据（`i`不同）
3. **同步执行**：Warp 内的 Thread 天然同步，无需`__syncthreads()`

### SIMT vs SIMD：关键差异

**术语**：SIMD (Single Instruction, Multiple Data，单指令多数据)

- 用于 CPU 并行计算领域
- **Single Instruction**：一条指令
- **Multiple Data**：多数据

**CPU 的 SIMD（如 AVX-512）**：

```c
// CPU SIMD：显式向量化
__m512 a = _mm512_load_ps(&A[i]);    // 加载16个float
__m512 b = _mm512_load_ps(&B[i]);
__m512 c = _mm512_add_ps(a, b);       // 16个加法
_mm512_store_ps(&C[i], c);

特点：
- 固定宽度：16个float（512-bit / 32-bit）
- 显式编程：程序员手动调用intrinsic
- 无分支：所有通道必须执行相同操作
```

**GPU 的 SIMT**：

```cuda
// GPU SIMT：隐式并行
C[i] = A[i] + B[i];

特点：
- 灵活宽度：硬件决定Warp大小（32）
- 隐式并行：编译器自动并行化
- 可分支：Thread可以走不同路径（有代价）
```

**对比表**：

| 特性         | CPU SIMD               | GPU SIMT                |
| ------------ | ---------------------- | ----------------------- |
| **向量宽度** | 固定（4/8/16）         | 固定 32（对程序员透明） |
| **编程方式** | 显式（需要 intrinsic） | 隐式（写标量代码）      |
| **分支处理** | 困难（需要 mask）      | 自动（硬件处理）        |
| **独立执行** | 不可以                 | 可以（用 Active Mask）  |
| **适用场景** | 规则数据并行           | 大规模数据并行          |

**为什么 SIMT 对 GPU 更好？**

1. **编程简单**：写标量代码，GPU 自动并行
2. **硬件灵活**：可以处理分支（虽然有代价）
3. **可扩展**：从 32 个 Thread 到数百万，代码不变

但代价是：**线程分歧会导致性能下降**。

---

## 3. 线程分歧：Warp 内的性能杀手

### 分歧的定义

**分歧**：同一 Warp 内的 Thread 执行不同的控制流路径。

### 完整的分歧示例

```cuda
__global__ void divergent(int* data, int threshold) {
    int i = threadIdx.x;
    int value = data[i];

    if (value > threshold) {
        // 路径A：复杂计算
        for (int j = 0; j < 100; j++) {
            value = value * value + j;
        }
    } else {
        // 路径B：简单计算
        value = value + 1;
    }

    data[i] = value;
}
```

假设输入数据：

```
Block有256个Thread = 8个Warp

Warp 0的数据：
Thread 0-15: value > threshold (走路径A)
Thread 16-31: value <= threshold (走路径B)
```

### 硬件如何执行分歧

**Warp 0 的执行时序**：

```
Cycle 0: 评估条件 (value > threshold)
  所有32个Thread执行比较
  硬件生成Active Mask:
    Mask = 0000000000000000111111111111111
           └──── 16个0 ───┘└─── 16个1 ───┘
            (Thread 16-31)  (Thread 0-15)

Cycle 1-100: 执行路径A (value > threshold为true)
  Active Mask = 1111111111111111000000000000000
  只有前16个Thread活跃
  后16个Thread IDLE（等待）

  执行100次循环：
    value = value * value + j

Cycle 101: 执行路径B (value > threshold为false)
  Active Mask = 0000000000000000111111111111111
  只有后16个Thread活跃
  前16个Thread IDLE（等待）

  执行1次计算：
    value = value + 1

Cycle 102: 合并路径
  Active Mask = 1111111111111111111111111111111
  所有32个Thread继续执行
    data[i] = value
```

**关键点**：

- 两个路径**串行执行**，不是并行
- IDLE 的 Thread 占用硬件资源但不工作
- 总时间 = 路径 A 时间 + 路径 B 时间

### 性能损失的精确计算

**场景分析**：

| 场景                | 路径 A 的 Thread 数 | 路径 B 的 Thread 数 | 执行时间   | 有效利用率 |
| ------------------- | ------------------- | ------------------- | ---------- | ---------- |
| **无分歧（全 A）**  | 32                  | 0                   | 100 cycles | 100%       |
| **无分歧（全 B）**  | 0                   | 32                  | 1 cycle    | 100%       |
| **分歧（16A+16B）** | 16                  | 16                  | 101 cycles | 50.2%      |
| **分歧（31A+1B）**  | 31                  | 1                   | 101 cycles | 31.2%      |
| **最坏（1A+31B）**  | 1                   | 31                  | 101 cycles | 3.5%       |

**计算公式**：

```
有效利用率 = 实际工作量 / (Warp大小 × 总时间)

分歧（16A+16B）：
  工作量 = 16×100 + 16×1 = 1616 Thread-cycles
  总时间 = 101 cycles
  利用率 = 1616 / (32 × 101) = 50.2%
```

### 实际测试（H100 GPU）

测试代码：

```cuda
// 版本1：有分歧
__global__ void withDivergence(float* data, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float value = data[i];
        if (value > 0.5f) {
            // 50%的Thread走这里
            for (int j = 0; j < 100; j++) {
                value = sqrtf(value) + j * 0.01f;
            }
        } else {
            // 50%的Thread走这里
            value = value + 1.0f;
        }
        data[i] = value;
    }
}

// 版本2：无分歧（数据重组后）
__global__ void noDivergence(float* data, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float value = data[i];
        // 现在所有Thread走相同路径
        for (int j = 0; j < 100; j++) {
            value = sqrtf(value) + j * 0.01f;
        }
        data[i] = value;
    }
}
```

**测试结果**（N=1,048,576，Block=256）：

```
有分歧版本：15.2 ms
无分歧版本：7.8 ms
加速比：1.95×
```

---

## 4. 避免分歧的策略

### 策略 1：重组数据以减少分歧

**原理**：让相同类型的数据聚集在一起，使得同一 Warp 内的 Thread 走相同路径。

**差的做法**：

```cuda
// 数据交替出现
float data[1024] = {
    0.1, 0.9,  // Thread 0走B，Thread 1走A
    0.2, 0.8,  // Thread 2走B，Thread 3走A
    ...
};

执行时：
Warp 0: 16个Thread走A，16个Thread走B（分歧严重）
```

**好的做法**：

```cuda
// 先小后大
float data[1024] = {
    0.1, 0.2, 0.3, ..., 0.5,  // 前512个都 <= 0.5
    0.6, 0.7, 0.8, ..., 1.0   // 后512个都 > 0.5
};

执行时：
Warp 0-15: 所有Thread走B（无分歧）
Warp 16-31: 所有Thread走A（无分歧）
```

**适用场景**：

- 排序后的数据
- 分类后的数据
- 预处理阶段可以重组数据的情况

### 策略 2：使用 Warp-level 原语

**问题场景**：Warp 内的 Thread 需要协作求和

**差的做法**：

```cuda
__global__ void badReduce(float* input, float* output) {
    __shared__ float temp[256];
    int tid = threadIdx.x;

    temp[tid] = input[tid];
    __syncthreads();

    // 只有Thread 0工作
    if (tid == 0) {
        float sum = 0;
        for (int i = 0; i < 256; i++) {
            sum += temp[i];
        }
        output[0] = sum;
    }
}

问题：
- Thread 1-31完全空闲（IDLE）
- Warp 0利用率：1/32 = 3.1%
```

**好的做法**（使用 Warp Shuffle）：

```cuda
__global__ void goodReduce(float* input, float* output) {
    int tid = threadIdx.x;
    float value = input[tid];

    // Warp内归约（无分歧，无Shared Memory）
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }

    // 每个Warp的第一个Thread写出结果
    if (tid % 32 == 0) {
        output[tid / 32] = value;
    }
}

优势：
- 所有32个Thread都参与
- Warp利用率：100%
- 无需Shared Memory
- 无需__syncthreads()
```

**Warp Shuffle 的执行过程**：

```
初始（Warp 0）：
Thread 0: value=1,  Thread 1: value=2,  ..., Thread 31: value=32

轮1 (offset=16):
Thread 0: value += Thread 16的value → value = 1+17 = 18
Thread 1: value += Thread 17的value → value = 2+18 = 20
...
Thread 15: value += Thread 31的value → value = 16+32 = 48

轮2 (offset=8):
Thread 0: value += Thread 8的value
Thread 1: value += Thread 9的value
...

...

轮5 (offset=1):
Thread 0: value += Thread 1的value → 最终和

最终：Thread 0有Warp的总和
```

### 策略 3：计算替代分支

**适用条件**：当计算成本 < 分支成本时

```cuda
// 差的做法（有分歧）
if (x > 0) {
    y = sqrtf(x);
} else {
    y = 0;
}

// 好的做法（无分歧，但多算了）
float result_a = sqrtf(fabs(x));  // 所有Thread都算
float result_b = 0;
y = (x > 0) ? result_a : result_b;  // 用三元运算符选择
```

**注意**：只有当分支内计算很简单时才有效

- ✓ 简单算术：1-2 个指令
- ✗ 复杂循环：100+个指令

### 策略 4：使用断言减少分支

```cuda
// 如果能保证数据范围，告诉编译器
__global__ void kernel(float* data) {
    int i = threadIdx.x;
    float value = data[i];

    // 断言：value总是正数
    assert(value > 0);

    // 编译器可能优化掉if
    float result = sqrtf(value);
}
```

---

## 5. Warp 调度：SM 内的并发机制

### 为什么需要多个 Warp？

回顾一个事实：内存访问延迟很高

```
全局内存访问延迟：400-800 cycles
计算指令延迟：4-8 cycles

如果SM只有1个Warp：
  发射内存加载 → 等待400 cycles → 继续计算
  GPU利用率：4 / 404 = 1%
```

**解决方案**：让 SM 同时驻留多个 Warp

```
SM有64个Warp槽位：

Warp 0: 发射内存加载 → 进入等待状态
        ↓（立即切换，0开销）
Warp 1: 发射计算指令 → 执行
        ↓
Warp 2: 发射计算指令 → 执行
        ↓
...
        ↓（400 cycles后）
Warp 0: 内存数据到达 → 恢复执行

理想情况：400 / 4 = 100个Warp能完全隐藏延迟
实际H100：每个SM最多64个Warp
```

### SM 的 Warp 池结构

**H100 SM 的简化结构**：

```
┌─────────────────────────────────────────────┐
│  Warp Scheduler × 4（4个独立调度器）        │
├─────────────────────────────────────────────┤
│  Warp池（最多64个Warp驻留）                 │
│  ┌─────┬─────┬─────┬─────┬─────┬─────┐     │
│  │Warp │Warp │Warp │Warp │Warp │Warp │     │
│  │  0  │  1  │  2  │  3  │  4  │ ... │     │
│  │Ready│Stall│Ready│Stall│Ready│Stall│     │
│  └─────┴─────┴─────┴─────┴─────┴─────┘     │
├─────────────────────────────────────────────┤
│  执行单元                                    │
│  ┌──────────────┐  ┌──────────────┐        │
│  │ FP32/INT32   │  │ Tensor Core  │        │
│  │ × 128个      │  │ × 4个        │        │
│  └──────────────┘  └──────────────┘        │
└─────────────────────────────────────────────┘
```

### Warp 的状态

一个 Warp 在 SM 上的生命周期：

```
1. Ready（就绪）：
   ├─ 所有操作数已准备好
   ├─ 需要的执行单元空闲
   └─ 无同步障碍

2. Stall（等待）：
   ├─ 内存访问未完成
   ├─ 等待__syncthreads()
   ├─ 数据依赖未解决
   └─ 执行单元忙碌

3. Executing（执行中）：
   └─ 占用执行单元，1-多个cycle
```

### 调度器的选择算法

**每个 cycle，调度器做什么？**

```python
# 伪代码：Warp调度
def schedule_warp():
    ready_warps = []

    # 步骤1：筛选Ready状态的Warp
    for warp in warp_pool:
        if warp.state == READY:
            if check_resources_available(warp):
                ready_warps.append(warp)

    if len(ready_warps) == 0:
        return None  # 所有Warp都在Stall，SM空闲

    # 步骤2：按优先级排序
    # 考虑因素：
    # - 指令类型（内存指令优先级低）
    # - 等待时间（长时间等待的优先）
    # - 公平性（轮询）

    ready_warps.sort(key=lambda w: w.priority)

    # 步骤3：发射指令
    selected = ready_warps[0]
    issue_instruction(selected)

    return selected
```

### 实际调度示例

假设 SM 上有 8 个 Warp：

```
Cycle 0:
Warp 0: Ready  （执行FMA指令）
Warp 1: Ready  （执行FMA指令）
Warp 2: Stall  （等待内存）
Warp 3: Ready  （执行ADD指令）
Warp 4-7: Stall

Scheduler选择: Warp 0（轮询，最早Ready）
发射: FMA指令（4 cycle延迟）

Cycle 1:
Warp 0: Stall  （FMA流水线中，还需3 cycles）
Warp 1: Ready
Warp 2: Stall
Warp 3: Ready

Scheduler选择: Warp 1
发射: FMA指令

Cycle 2:
Warp 0: Stall  （还需2 cycles）
Warp 1: Stall  （还需3 cycles）
Warp 2: Stall
Warp 3: Ready

Scheduler选择: Warp 3
发射: ADD指令（1 cycle延迟）

Cycle 3:
Warp 0: Stall  （还需1 cycle）
Warp 1: Stall
Warp 2: Ready  （内存数据返回！）
Warp 3: Ready

Scheduler选择: Warp 2
发射: MUL指令

Cycle 4:
Warp 0: Ready  （FMA完成）
Warp 1: Stall
Warp 2: Ready
Warp 3: Ready

Scheduler选择: Warp 0（轮询回到Warp 0）
```

**关键观察**：

- 调度器每个 cycle 都在寻找 Ready 的 Warp
- 切换 Warp 完全没有开销（硬件实现）
- 多个 Warp 隐藏了各自的延迟

---

## 6. Occupancy：资源与并发的权衡

### Occupancy 的定义

```
Occupancy = 活跃Warp数 / SM最大Warp数

H100: 最大64个Warp/SM
Occupancy范围: 0% - 100%
```

### 为什么 Occupancy 重要？

**延迟隐藏的需求**：

```
内存访问延迟: 400 cycles
指令执行时间: 4 cycles

理想情况（完全隐藏延迟）：
  需要: 400 / 4 = 100个Warp

实际H100:
  最多: 64个Warp/SM
  能隐藏: 64 × 4 = 256 cycles
  隐藏率: 256 / 400 = 64%

如果Occupancy只有50%（32个Warp）:
  能隐藏: 32 × 4 = 128 cycles
  隐藏率: 128 / 400 = 32%
```

低 Occupancy 意味着：更多的 Stall 时间，更低的 SM 利用率。

### 影响 Occupancy 的资源

**H100 每个 SM 的资源总量**：

| 资源          | 总量      | 每 Block 限制 | 每 Thread 限制 |
| ------------- | --------- | ------------- | -------------- |
| 寄存器        | 65,536 个 | 无            | 255 个         |
| Shared Memory | 228 KB    | 228 KB        | 无             |
| 线程数        | 2048 个   | 1024 个       | -              |
| Block 数      | 32 个     | -             | -              |
| **Warp 数**   | **64 个** | **32 个**     | **-**          |

### Occupancy 的精确计算

**计算公式**：

```
活跃Warp数 = min(
    ⌊寄存器总量 / (寄存器/Thread × 32)⌋,
    ⌊Shared Memory总量 / Shared Memory/Block⌋ × Warp/Block,
    ⌊线程数总量 / Thread/Block⌋ × Warp/Block,
    Block数限制 × Warp/Block
)

Occupancy = 活跃Warp数 / 64
```

### 完整计算示例

**给定配置**：

```
Block大小: 256 threads (8个Warp)
寄存器使用: 40个/thread
Shared Memory: 8 KB/block
```

**步骤 1：寄存器限制**

```
每个Warp需要寄存器:
  40 reg/thread × 32 thread/warp = 1280 registers

最多能驻留的Warp数:
  65536 / 1280 = 51.2 → 51 warps
```

**步骤 2：Shared Memory 限制**

```
每个Block需要: 8 KB
最多能驻留的Block数:
  228 KB / 8 KB = 28.5 → 28 blocks

每个Block有8个Warp:
  28 blocks × 8 warps/block = 224 warps

224 > 64，所以Shared Memory不是瓶颈
```

**步骤 3：线程数限制**

```
每个Block: 256 threads
最多能驻留的Block数:
  2048 / 256 = 8 blocks

Warp数:
  8 blocks × 8 warps/block = 64 warps
```

**步骤 4：Block 数限制**

```
最多32个Block/SM
每个Block 8个Warp:
  32 × 8 = 256 warps

256 > 64，不是瓶颈
```

**结论**：

```
瓶颈: 寄存器（51 warps）
Occupancy = 51 / 64 = 79.7%
```

### 可视化不同配置的对比

**配置对比**（相同 kernel，不同 Block 大小）：

| Block 大小 | Warp/Block | 寄存器限制 | 线程数限制         | 瓶颈   | Occupancy |
| ---------- | ---------- | ---------- | ------------------ | ------ | --------- |
| 128        | 4          | 51 warps   | 16 blocks × 4 = 64 | 寄存器 | 79.7%     |
| 256        | 8          | 51 warps   | 8 blocks × 8 = 64  | 寄存器 | 79.7%     |
| 512        | 16         | 51 warps   | 4 blocks × 16 = 64 | 寄存器 | 79.7%     |

**如果减少寄存器到 32 个/thread**：

| Block 大小 | Warp/Block | 寄存器限制 | 线程数限制 | 瓶颈 | Occupancy |
| ---------- | ---------- | ---------- | ---------- | ---- | --------- |
| 128        | 4          | 64 warps   | 64 warps   | 平衡 | 100%      |
| 256        | 8          | 64 warps   | 64 warps   | 平衡 | 100%      |
| 512        | 16         | 64 warps   | 64 warps   | 平衡 | 100%      |

---

## 7. Occupancy 优化策略

### 策略 1：减少寄存器使用

**方法 1：使用`__launch_bounds__`**

```cuda
// 告诉编译器：每个Block 256线程，至少8个Block/SM
__global__ void __launch_bounds__(256, 8)
myKernel() {
    // 编译器会限制寄存器使用以达到目标Occupancy
}

计算目标：
  8 blocks × 256 threads = 2048 threads
  2048 / 32 = 64 warps （100% Occupancy）

  寄存器预算:
    65536 / 64 = 1024 registers/warp
    1024 / 32 = 32 registers/thread
```

**方法 2：手动减少局部变量**

```cuda
// 差的做法：大量中间变量
__global__ void badKernel() {
    float a1, a2, a3, ..., a20;  // 可能用50+个寄存器
    a1 = compute1();
    a2 = compute2();
    ...
}

// 好的做法：重用变量
__global__ void goodKernel() {
    float temp;  // 只用几个寄存器
    temp = compute1();
    result1 = process(temp);

    temp = compute2();  // 重用
    result2 = process(temp);
}
```

### 策略 2：调整 Shared Memory 使用

```cuda
// 如果Shared Memory是瓶颈

// 差的做法：分配过大
__shared__ float buffer[4096];  // 16 KB

最多Block数: 228 KB / 16 KB = 14 blocks
Occupancy可能很低

// 好的做法：精确计算需要的大小
__shared__ float buffer[1024];  // 4 KB

最多Block数: 228 KB / 4 KB = 57 blocks（足够）
```

### 策略 3：选择合适的 Block 大小

**实验**（相同 kernel，不同 Block 大小）：

```
寄存器使用: 40个/thread
Shared Memory: 0

Block=128:
  最多Block数: 2048 / 128 = 16
  寄存器限制: 51 warps
  实际Warp数: min(51, 16×4) = 51
  Occupancy: 79.7%

Block=256:
  最多Block数: 2048 / 256 = 8
  寄存器限制: 51 warps
  实际Warp数: min(51, 8×8) = 51
  Occupancy: 79.7%

Block=512:
  最多Block数: 2048 / 512 = 4
  寄存器限制: 51 warps
  实际Warp数: min(51, 4×16) = 51
  Occupancy: 79.7%

结论：这个kernel的Block大小对Occupancy影响不大
      瓶颈在寄存器
```

---

## 8. 反直觉的真相：高 Occupancy ≠ 高性能

### 矩阵乘法实验

三个版本的对比：

**版本 1：无 Shared Memory**

```cuda
__global__ void matmul_v1(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

配置: Block(16, 16)
寄存器: ~10个/thread
Shared Memory: 0
Occupancy: 100%
性能: 5 TFLOPS
瓶颈: 全局内存带宽
```

**版本 2：使用 Shared Memory**

```cuda
#define TILE 16

__global__ void matmul_v2(float* A, float* B, float* C, int N) {
    __shared__ float As[TILE][TILE];  // 1 KB
    __shared__ float Bs[TILE][TILE];  // 1 KB
    // 总共 2 KB Shared Memory

    // ... tile加载和计算 ...
}

配置: Block(16, 16)
寄存器: ~15个/thread
Shared Memory: 2 KB/block
Occupancy: 100%
性能: 15 TFLOPS (3×提升!)
瓶颈: 计算
```

**版本 3：更大的 Tile**

```cuda
#define TILE 32

__global__ void matmul_v3(float* A, float* B, float* C, int N) {
    __shared__ float As[TILE][TILE+1];  // 4.1 KB
    __shared__ float Bs[TILE][TILE+1];  // 4.1 KB
    // 总共 8.2 KB Shared Memory

    // ... tile加载和计算 ...
}

配置: Block(32, 32) = 1024 threads
寄存器: ~20个/thread
Shared Memory: 8.2 KB/block
Occupancy: 约50%
性能: 20 TFLOPS (4×提升!)
瓶颈: 计算
```

### 为什么低 Occupancy 更快？

**关键因素对比**：

| 因素             | v1 (100% Occ)   | v2 (100% Occ)   | v3 (50% Occ)    |
| ---------------- | --------------- | --------------- | --------------- |
| **数据重用**     | 无              | 16×             | 32×             |
| **全局内存访问** | 每个元素读 N 次 | 每个元素读 1 次 | 每个元素读 1 次 |
| **寄存器压力**   | 低              | 中              | 高              |
| **缓存命中率**   | 低              | 中              | 高              |

**v3 虽然 Occupancy 低，但：**

1. 数据重用更好：32×32 tile，每个数据被重用 32 次
2. 全局内存带宽需求降低 32×
3. 更多寄存器可用：减少寄存器溢出
4. 更好的缓存命中率：活跃 Warp 少，每个 Warp 的缓存份额更大

### Occupancy 的经验法则

| Occupancy 范围 | 含义         | 建议                       |
| -------------- | ------------ | -------------------------- |
| **< 25%**      | 可能有问题   | 检查资源使用，是否过度优化 |
| **25-50%**     | 计算密集型   | 如果性能好，可接受         |
| **50-75%**     | 平衡型       | 通常的优化目标             |
| **75-100%**    | 内存延迟受限 | 需要高 Occupancy 隐藏延迟  |

**优化流程**：

```
1. 识别瓶颈（使用profiler）
   ↓
2. 如果瓶颈是内存延迟: → 提升Occupancy（减少寄存器/Shared Memory使用）
   ↓
3. 如果瓶颈是带宽: → 优化内存访问模式（可能降低Occupancy）
   ↓
4. 如果瓶颈是计算: → 优化算法（Occupancy可能不重要）
```

---

## 9. 实战案例：Reduction 优化

### 版本 1：朴素实现

```cuda
__global__ void reduce_v1(float* input, float* output, int N) {
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < N) ? input[i] : 0;
    __syncthreads();

    // 树形归约（有严重分歧）
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
轮1 (s=1): tid % 2 == 0
  Warp 0: Thread 0,2,4,...,30活跃（16个）
  分歧：50%利用率

轮2 (s=2): tid % 4 == 0
  Warp 0: Thread 0,4,8,...,28活跃（8个）
  分歧：25%利用率

...

轮8 (s=128): tid % 256 == 0
  Warp 0: Thread 0活跃（1个）
  分歧：3.1%利用率

性能: ~50%理论峰值
```

### 版本 2：消除分歧

```cuda
__global__ void reduce_v2(float* input, float* output, int N) {
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < N) ? input[i] : 0;
    __syncthreads();

    // 相邻Thread协作（减少分歧）
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
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
轮1 (s=128): tid < 128
  Warp 0-3: 所有Thread活跃（无分歧）
  Warp 4-7: 所有Thread IDLE

轮2 (s=64): tid < 64
  Warp 0-1: 所有Thread活跃（无分歧）
  Warp 2-7: IDLE

轮3 (s=32): tid < 32
  Warp 0: 所有Thread活跃（无分歧）
  Warp 1-7: IDLE

轮4 (s=16): tid < 16
  Warp 0: 一半Thread活跃（开始分歧）

性能: ~85%理论峰值
提升: 1.7×
```

### 版本 3：Warp Shuffle

```cuda
__device__ float warpReduce(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void reduce_v3(float* input, float* output, int N) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float val = (i < N) ? input[i] : 0;

    // Warp内归约（无同步开销，无分歧）
    val = warpReduce(val);

    // 每个Warp的第一个Thread写到Shared Memory
    __shared__ float warpSums[8];  // 256/32 = 8个Warp
    if (tid % 32 == 0) {
        warpSums[tid / 32] = val;
    }
    __syncthreads();

    // 第一个Warp归约所有Warp的结果
    if (tid < 8) {
        val = warpSums[tid];
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
1. 无Shared Memory bank conflict（概念留给内存篇）
2. 无分支分化（Warp内天然同步）
3. 无__syncthreads()开销（除了一次）

性能: ~98%理论峰值
总提升: 2.5×
```

---

## 10. 总结与展望

### 核心概念回顾

**Warp：GPU 执行的真实单位**

```
软件视角              硬件实现
  Thread      →        Warp (32 Thread)
  独立计算     →        SIMT执行
```

**线程分歧：性能杀手**

```
分歧成本 = 串行执行不同路径
避免方法：
  1. 重组数据
  2. Warp-level原语
  3. 计算替代分支
```

**Warp 调度：延迟隐藏**

```
SM驻留多个Warp → 切换无开销 → 隐藏内存延迟
```

**Occupancy：资源权衡**

```
Occupancy = 活跃Warp数 / 64
受限因素：寄存器、Shared Memory、线程数

关键洞察：高Occupancy ≠ 高性能
```

### 设计原则

1. **Block 大小必须是 32 的倍数**：对应 Warp 大小
2. **避免线程分歧**：同一 Warp 内 Thread 走相同路径
3. **平衡 Occupancy 和资源使用**：不盲目追求 100%
4. **用 Warp-level 原语**：`__shfl_*`比 Shared Memory 更快

### 性能优化 checklist

```
✓ Block是32的倍数
✓ 识别并消除线程分歧
✓ 计算Occupancy，识别资源瓶颈
✓ 考虑数据重用 vs Occupancy的权衡
✓ 使用Warp Shuffle替代Shared Memory（当适用时）
```

### 下一节

**深入内存模型** → 《CUDA 内存模型：从 Global 到 Shared Memory》

- 合并访问（Coalesced Access）
- Bank Conflict 详解
- L1/L2 Cache 利用
