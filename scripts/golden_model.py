import math


def bit_packing_golden_model():
    print("=== 纯 Python 版 PackKV 位打包参考模型 ===")

    # 假设这是经过量化和重排后，某一个 Pack(组) 里的 16 个整数
    # 我们可以看到这组数据同质性很高
    pack_data = [4, 5, 4, 6, 5, 4, 4, 5, 6, 5, 4, 4, 5, 6, 4, 5]
    print(f"1. 原始输入数据 (16个元素): {pack_data}")

    # ---------------------------------------------------------
    # 步骤 1：提取极值与计算动态位宽
    # ---------------------------------------------------------
    min_val = min(pack_data)
    max_val = max(pack_data)
    data_range = max_val - min_val

    # 计算位宽公式: ceil(log2(range + 1))
    bit_width = math.ceil(math.log2(data_range + 1)) if data_range > 0 else 0

    print(f"2. 提取特征 -> 最小值: {min_val}, 最大值: {max_val}, 极差: {data_range}")
    print(f"3. 动态分配位宽: {bit_width} bits")

    # ---------------------------------------------------------
    # 步骤 2：计算偏移量 (Payload)
    # ---------------------------------------------------------
    offsets = [x - min_val for x in pack_data]
    print(f"4. 扣除基准值后的偏移量: {offsets}")

    # ---------------------------------------------------------
    # 步骤 3：模拟底层硬件寄存器的位拼接 (Bit-packing)
    # ---------------------------------------------------------
    # 在实际系统(C++或RTL中)，我们通常用一个 32位 或 64位的无符号整数作为容器
    hardware_register = 0  # 初始状态全 0

    print("\n5. 开始将数据压入模拟寄存器...")
    for i, offset in enumerate(offsets):
        # 核心硬件操作：将数据左移到它该去的位置，然后用"按位或"存入寄存器
        # 相当于在硬件电路上把线(wires)拼接起来
        shifted_val = offset << (i * bit_width)
        hardware_register = hardware_register | shifted_val

        # 打印二进制状态，观察寄存器是如何被一点点填满的
        # :032b 表示格式化为 32 位二进制，前面补零
        print(
            f"  压入第 {i:2d} 个数 (值={offset}): 寄存器二进制 = {hardware_register:032b}"
        )

    print(f"\n6. 打包完成！最终的 32-bit 数据载荷 (十进制): {hardware_register}")

    # ---------------------------------------------------------
    # 步骤 4：模拟解压 (Unpacking)
    # ---------------------------------------------------------
    print("\n=== 开始解压测试 ===")
    restored_data = []
    # 制作一个掩码(Mask)，比如 bit_width=2 时，mask 就是二进制的 11 (即十进制3)
    mask = (1 << bit_width) - 1

    for i in range(len(pack_data)):
        # 核心硬件操作：右移到对应位置，然后用掩码(Mask)抠出我们需要的那几个 bit
        extracted_offset = (hardware_register >> (i * bit_width)) & mask
        # 加上之前存的元数据(最小值)
        original_val = extracted_offset + min_val
        restored_data.append(original_val)

    print(f"解压后的数据: {restored_data}")
    print(f"是否无损还原? {restored_data == pack_data}")


if __name__ == "__main__":
    bit_packing_golden_model()
