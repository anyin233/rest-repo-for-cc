"""
测试三种调度算法的性能
使用四峰分布数据集
"""

import numpy as np
import matplotlib.pyplot as plt
from scheduler import (
    Task, RoundRobinScheduler, ShortestQueueScheduler,
    ProbabilisticScheduler, DiscreteEventSimulator, calculate_metrics
)
from typing import List
import pandas as pd


def create_four_peak_distribution(size: int, seed: int = 42) -> np.ndarray:
    """创建四峰分布数据集

    四个峰的中位数分别为: 50, 100, 175, 375 秒
    使用高斯混合模型

    Args:
        size: 样本数量
        seed: 随机种子

    Returns:
        执行时间数组（秒）
    """
    np.random.seed(seed)

    # 定义四个峰
    peaks = [
        {'median': 50, 'std': 10, 'weight': 0.25},
        {'median': 100, 'std': 15, 'weight': 0.25},
        {'median': 175, 'std': 20, 'weight': 0.25},
        {'median': 375, 'std': 40, 'weight': 0.25},
    ]

    samples = []

    for peak in peaks:
        n_samples = int(size * peak['weight'])
        # 使用正态分布生成样本
        peak_samples = np.random.normal(peak['median'], peak['std'], n_samples)
        # 确保执行时间为正
        peak_samples = np.maximum(peak_samples, 1.0)
        samples.extend(peak_samples)

    # 如果由于四舍五入导致样本数不足，补充一些
    while len(samples) < size:
        peak = np.random.choice(peaks)
        sample = np.random.normal(peak['median'], peak['std'])
        samples.append(max(sample, 1.0))

    # 随机打乱
    samples = np.array(samples[:size])
    np.random.shuffle(samples)

    return samples


def visualize_distribution(execution_times: np.ndarray, filename: str = 'distribution.png'):
    """可视化执行时间分布"""
    plt.figure(figsize=(12, 6))

    # 直方图
    plt.subplot(1, 2, 1)
    plt.hist(execution_times, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Execution Time (seconds)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Task Execution Times')
    plt.axvline(50, color='r', linestyle='--', alpha=0.5, label='Peak 1: 50s')
    plt.axvline(100, color='g', linestyle='--', alpha=0.5, label='Peak 2: 100s')
    plt.axvline(175, color='b', linestyle='--', alpha=0.5, label='Peak 3: 175s')
    plt.axvline(375, color='m', linestyle='--', alpha=0.5, label='Peak 4: 375s')
    plt.legend()
    plt.grid(alpha=0.3)

    # 累积分布
    plt.subplot(1, 2, 2)
    sorted_times = np.sort(execution_times)
    cumulative = np.arange(1, len(sorted_times) + 1) / len(sorted_times)
    plt.plot(sorted_times, cumulative)
    plt.xlabel('Execution Time (seconds)')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution Function')
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Distribution visualization saved to {filename}")
    plt.close()


def generate_tasks(execution_times: np.ndarray, arrival_rate: float = 1.0) -> List[Task]:
    """生成任务列表

    Args:
        execution_times: 执行时间数组
        arrival_rate: 任务到达率（每秒）

    Returns:
        任务列表
    """
    tasks = []
    current_time = 0.0

    for i, exec_time in enumerate(execution_times):
        # 使用泊松过程生成到达时间间隔
        inter_arrival = np.random.exponential(1.0 / arrival_rate)
        current_time += inter_arrival

        task = Task(
            task_id=i,
            execution_time=exec_time,
            arrival_time=current_time
        )
        tasks.append(task)

    return tasks


def run_experiment(num_nodes: int, num_tasks: int, arrival_rate: float, seed: int = 42):
    """运行实验比较三种调度算法"""

    print(f"\n{'='*80}")
    print(f"实验配置:")
    print(f"  节点数: {num_nodes}")
    print(f"  任务数: {num_tasks}")
    print(f"  到达率: {arrival_rate} 任务/秒")
    print(f"  随机种子: {seed}")
    print(f"{'='*80}\n")

    # 生成四峰分布数据集
    print("生成四峰分布数据集...")
    execution_times = create_four_peak_distribution(num_tasks, seed=seed)

    print(f"数据集统计:")
    print(f"  均值: {np.mean(execution_times):.2f} 秒")
    print(f"  中位数: {np.median(execution_times):.2f} 秒")
    print(f"  标准差: {np.std(execution_times):.2f} 秒")
    print(f"  最小值: {np.min(execution_times):.2f} 秒")
    print(f"  最大值: {np.max(execution_times):.2f} 秒")

    # 可视化分布
    visualize_distribution(execution_times)

    # 生成任务
    print("\n生成任务列表...")
    tasks_template = generate_tasks(execution_times, arrival_rate)

    # 定义三种调度器
    schedulers = {
        'Round Robin': RoundRobinScheduler(num_nodes),
        'Shortest Queue': ShortestQueueScheduler(num_nodes, variance_weight=0.5),
        'Probabilistic': ProbabilisticScheduler(num_nodes, num_samples=1),
    }

    results = {}

    # 对每种调度器运行仿真
    for name, scheduler in schedulers.items():
        print(f"\n运行 {name} 调度器...")

        # 创建任务副本（每个调度器使用相同的任务）
        tasks = [Task(t.task_id, t.execution_time, t.arrival_time) for t in tasks_template]

        # 创建仿真器
        sim = DiscreteEventSimulator(scheduler)

        # 添加所有任务到达事件
        for task in tasks:
            sim.add_arrival_event(task)

        # 运行仿真
        task_metrics = sim.run()

        # 计算性能指标
        metrics = calculate_metrics(task_metrics)
        results[name] = metrics

        print(f"  完成仿真，处理了 {len(task_metrics)} 个任务")

    return results, execution_times


def print_results(results: dict):
    """打印结果对比"""
    print(f"\n{'='*80}")
    print("性能指标对比")
    print(f"{'='*80}\n")

    # 创建DataFrame方便对比
    df = pd.DataFrame(results).T

    # 按列显示
    metrics_to_show = [
        'avg_response_time',
        'median_response_time',
        'p95_response_time',
        'p99_response_time',
        'avg_waiting_time',
        'median_waiting_time',
        'std_response_time',
        'max_response_time',
    ]

    print("响应时间指标 (秒):")
    print("-" * 80)
    for metric in metrics_to_show:
        if metric in df.columns:
            print(f"\n{metric}:")
            for scheduler in df.index:
                value = df.loc[scheduler, metric]
                print(f"  {scheduler:20s}: {value:10.2f}")

    # 找出最佳调度器
    print(f"\n{'='*80}")
    print("最佳调度器（基于各项指标）:")
    print(f"{'='*80}\n")

    for metric in ['avg_response_time', 'median_response_time', 'p95_response_time', 'p99_response_time']:
        if metric in df.columns:
            best = df[metric].idxmin()
            print(f"  {metric:30s}: {best} ({df.loc[best, metric]:.2f}s)")


def visualize_results(results: dict, filename: str = 'results_comparison.png'):
    """可视化结果对比"""
    metrics = ['avg_response_time', 'median_response_time', 'p95_response_time', 'p99_response_time']
    schedulers = list(results.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = [results[scheduler][metric] for scheduler in schedulers]

        bars = ax.bar(schedulers, values, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Time (seconds)')
        ax.set_title(metric.replace('_', ' ').title())
        ax.grid(alpha=0.3, axis='y')

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom')

        # 旋转x轴标签
        ax.tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nResults visualization saved to {filename}")
    plt.close()


def main():
    """主函数"""
    # 实验参数
    num_nodes = 4
    num_tasks = 1000
    arrival_rate = 0.5  # 每秒0.5个任务

    # 运行实验
    results, execution_times = run_experiment(num_nodes, num_tasks, arrival_rate)

    # 打印结果
    print_results(results)

    # 可视化结果
    visualize_results(results)

    print(f"\n{'='*80}")
    print("实验完成!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
