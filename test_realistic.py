"""
测试真实调度器实现
"""

import numpy as np
import matplotlib.pyplot as plt
from realistic_scheduler import (
    Task, TaskDistribution, RoundRobinScheduler, ShortestQueueScheduler,
    ProbabilisticScheduler, RealisticSimulator, calculate_metrics
)
from typing import List
import pandas as pd


def create_four_peak_task_distributions() -> List[TaskDistribution]:
    """创建四峰分布

    四个峰的中位数分别为: 50, 100, 175, 375 秒
    """
    peaks = [
        TaskDistribution.from_normal(mean=50, std=10),
        TaskDistribution.from_normal(mean=100, std=15),
        TaskDistribution.from_normal(mean=175, std=20),
        TaskDistribution.from_normal(mean=375, std=40),
    ]
    return peaks


def generate_tasks(num_tasks: int, arrival_rate: float, task_distributions: List[TaskDistribution]) -> List[Task]:
    """生成任务列表

    Args:
        num_tasks: 任务数量
        arrival_rate: 到达率（每秒）
        task_distributions: 任务分布列表

    Returns:
        任务列表
    """
    tasks = []
    current_time = 0.0

    for i in range(num_tasks):
        # 泊松到达
        inter_arrival = np.random.exponential(1.0 / arrival_rate)
        current_time += inter_arrival

        # 随机选择一个峰
        dist = np.random.choice(task_distributions)

        # 采样实际执行时间
        actual_time = dist.sample()

        task = Task(
            task_id=i,
            distribution=dist,
            actual_execution_time=actual_time,
            arrival_time=current_time
        )
        tasks.append(task)

    return tasks


def visualize_task_distribution(tasks: List[Task], filename: str = 'realistic_distribution.png'):
    """可视化任务分布"""
    execution_times = [task.actual_execution_time for task in tasks]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 直方图
    axes[0].hist(execution_times, bins=50, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Execution Time (seconds)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Task Execution Times')
    axes[0].axvline(50, color='r', linestyle='--', alpha=0.5, label='Peak 1: 50s')
    axes[0].axvline(100, color='g', linestyle='--', alpha=0.5, label='Peak 2: 100s')
    axes[0].axvline(175, color='b', linestyle='--', alpha=0.5, label='Peak 3: 175s')
    axes[0].axvline(375, color='m', linestyle='--', alpha=0.5, label='Peak 4: 375s')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 累积分布
    sorted_times = np.sort(execution_times)
    cumulative = np.arange(1, len(sorted_times) + 1) / len(sorted_times)
    axes[1].plot(sorted_times, cumulative)
    axes[1].set_xlabel('Execution Time (seconds)')
    axes[1].set_ylabel('Cumulative Probability')
    axes[1].set_title('Cumulative Distribution Function')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Task distribution saved to {filename}")
    plt.close()


def run_experiment(num_nodes: int, num_tasks: int, arrival_rate: float, seed: int = 42):
    """运行实验"""
    np.random.seed(seed)

    print(f"\n{'='*80}")
    print(f"真实调度器实验配置:")
    print(f"  节点数: {num_nodes}")
    print(f"  任务数: {num_tasks}")
    print(f"  到达率: {arrival_rate} 任务/秒")
    print(f"  随机种子: {seed}")
    print(f"{'='*80}\n")

    # 创建四峰分布
    print("创建四峰分布...")
    task_distributions = create_four_peak_task_distributions()

    print("四峰分布统计:")
    for i, dist in enumerate(task_distributions):
        print(f"  峰 {i+1}: 均值={dist.mean:.2f}s, 标准差={np.sqrt(dist.variance):.2f}s, "
              f"中位数={dist.quantiles[3]:.2f}s")

    # 生成任务
    print("\n生成任务列表...")
    tasks_template = generate_tasks(num_tasks, arrival_rate, task_distributions)

    execution_times = [t.actual_execution_time for t in tasks_template]
    print(f"\n实际执行时间统计:")
    print(f"  均值: {np.mean(execution_times):.2f} 秒")
    print(f"  中位数: {np.median(execution_times):.2f} 秒")
    print(f"  标准差: {np.std(execution_times):.2f} 秒")
    print(f"  最小值: {np.min(execution_times):.2f} 秒")
    print(f"  最大值: {np.max(execution_times):.2f} 秒")

    # 可视化
    visualize_task_distribution(tasks_template)

    # 定义三种调度器
    schedulers = {
        'Round Robin': lambda: RoundRobinScheduler(num_nodes),
        'Shortest Queue': lambda: ShortestQueueScheduler(num_nodes, variance_weight=0.5),
        'Probabilistic': lambda: ProbabilisticScheduler(num_nodes),
    }

    results = {}

    # 对每种调度器运行仿真
    for name, scheduler_factory in schedulers.items():
        print(f"\n运行 {name} 调度器...")

        # 重新生成任务（使用相同的种子确保一致性）
        np.random.seed(seed)
        tasks = generate_tasks(num_tasks, arrival_rate, task_distributions)

        # 创建调度器和仿真器
        scheduler = scheduler_factory()
        sim = RealisticSimulator(scheduler)

        # 添加所有任务到达事件
        for task in tasks:
            sim.add_task_arrival(task)

        # 运行仿真
        completed_tasks = sim.run()

        # 计算性能指标
        metrics = calculate_metrics(completed_tasks)
        results[name] = metrics

        print(f"  完成 {len(completed_tasks)} 个任务")
        print(f"  平均响应时间: {metrics['avg_response_time']:.2f}s")
        print(f"  中位数响应时间: {metrics['median_response_time']:.2f}s")

    return results


def print_results(results: dict):
    """打印结果对比"""
    print(f"\n{'='*80}")
    print("性能指标对比")
    print(f"{'='*80}\n")

    df = pd.DataFrame(results).T

    metrics_to_show = [
        ('avg_response_time', '平均响应时间'),
        ('median_response_time', '中位数响应时间'),
        ('p95_response_time', 'P95响应时间'),
        ('p99_response_time', 'P99响应时间'),
        ('avg_waiting_time', '平均等待时间'),
        ('median_waiting_time', '中位数等待时间'),
        ('std_response_time', '响应时间标准差'),
        ('max_response_time', '最大响应时间'),
    ]

    for metric, label in metrics_to_show:
        if metric in df.columns:
            print(f"\n{label} ({metric}):")
            print("-" * 80)
            for scheduler in df.index:
                value = df.loc[scheduler, metric]
                print(f"  {scheduler:20s}: {value:10.2f}s")

    # 找出最佳调度器
    print(f"\n{'='*80}")
    print("最佳调度器（基于各项指标）:")
    print(f"{'='*80}\n")

    for metric, label in [
        ('avg_response_time', '平均响应时间'),
        ('median_response_time', '中位数响应时间'),
        ('p95_response_time', 'P95响应时间'),
        ('p99_response_time', 'P99响应时间')
    ]:
        if metric in df.columns:
            best = df[metric].idxmin()
            print(f"  {label:20s}: {best} ({df.loc[best, metric]:.2f}s)")


def visualize_results(results: dict, filename: str = 'realistic_comparison.png'):
    """可视化结果"""
    metrics = [
        ('avg_response_time', 'Average Response Time'),
        ('median_response_time', 'Median Response Time'),
        ('p95_response_time', 'P95 Response Time'),
        ('p99_response_time', 'P99 Response Time')
    ]
    schedulers = list(results.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx]
        values = [results[scheduler][metric] for scheduler in schedulers]

        bars = ax.bar(schedulers, values, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Time (seconds)')
        ax.set_title(title)
        ax.grid(alpha=0.3, axis='y')

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom')

        ax.tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nResults comparison saved to {filename}")
    plt.close()


def test_dynamic_queue_updates(num_nodes: int = 4, num_tasks: int = 20):
    """测试动态队列更新功能"""
    print(f"\n{'='*80}")
    print("测试动态队列更新功能")
    print(f"{'='*80}\n")

    np.random.seed(42)

    # 创建简单的任务分布
    task_dist = TaskDistribution.from_normal(mean=100, std=10)

    # 生成少量任务便于观察
    tasks = []
    for i in range(num_tasks):
        task = Task(
            task_id=i,
            distribution=task_dist,
            actual_execution_time=task_dist.sample(),
            arrival_time=i * 10.0  # 每10秒到达一个任务
        )
        tasks.append(task)

    # 测试Shortest Queue调度器
    print("测试 Shortest Queue 调度器的队列更新:")
    scheduler = ShortestQueueScheduler(num_nodes)
    sim = RealisticSimulator(scheduler)

    for task in tasks[:5]:  # 只添加前5个任务便于观察
        sim.add_task_arrival(task)

    print("\n添加5个任务后，各队列状态:")
    for i, queue in enumerate(scheduler.queues):
        print(f"  队列 {i}: 任务数={len(queue.tasks)}, "
              f"期望时间={queue.distribution.mean:.2f}s, "
              f"方差={queue.distribution.variance:.2f}")

    # 运行仿真
    for task in tasks:
        sim.add_task_arrival(task)

    completed = sim.run()

    print(f"\n完成 {len(completed)} 个任务")
    print("\n最终队列状态（应该全部清空）:")
    for i, queue in enumerate(scheduler.queues):
        print(f"  队列 {i}: 任务数={len(queue.tasks)}, "
              f"期望时间={queue.distribution.mean:.2f}s, "
              f"方差={queue.distribution.variance:.2f}")

    print("\n测试 Probabilistic 调度器的分位数更新:")
    np.random.seed(42)
    scheduler = ProbabilisticScheduler(num_nodes)
    sim = RealisticSimulator(scheduler)

    tasks = []
    for i in range(num_tasks):
        task = Task(
            task_id=i,
            distribution=task_dist,
            actual_execution_time=task_dist.sample(),
            arrival_time=i * 10.0
        )
        tasks.add(task)

    for task in tasks[:5]:
        sim.add_task_arrival(task)

    print("\n添加5个任务后，各队列分位数:")
    for i, queue in enumerate(scheduler.queues):
        print(f"  队列 {i}: P50={queue.distribution.quantiles[3]:.2f}s, "
              f"P95={queue.distribution.quantiles[6]:.2f}s")

    for task in tasks:
        sim.add_task_arrival(task)

    completed = sim.run()

    print(f"\n完成 {len(completed)} 个任务")
    print("\n最终队列分位数（应该接近零）:")
    for i, queue in enumerate(scheduler.queues):
        print(f"  队列 {i}: P50={queue.distribution.quantiles[3]:.2f}s, "
              f"P95={queue.distribution.quantiles[6]:.2f}s")


def main():
    """主函数"""
    # 基础实验
    num_nodes = 4
    num_tasks = 1000
    arrival_rate = 0.5

    results = run_experiment(num_nodes, num_tasks, arrival_rate)
    print_results(results)
    visualize_results(results)

    # 测试动态更新
    # test_dynamic_queue_updates()

    print(f"\n{'='*80}")
    print("实验完成!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
