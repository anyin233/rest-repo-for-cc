"""
高级分析脚本
对三种调度算法进行更全面的性能测试
"""

import numpy as np
import matplotlib.pyplot as plt
from scheduler import (
    Task, RoundRobinScheduler, ShortestQueueScheduler,
    ProbabilisticScheduler, DiscreteEventSimulator, calculate_metrics
)
from test_scheduler import create_four_peak_distribution, generate_tasks
import pandas as pd
from typing import Dict, List


def run_multiple_experiments(
    num_nodes_list: List[int],
    num_tasks: int,
    arrival_rate: float,
    num_runs: int = 5
) -> Dict:
    """运行多组实验，测试不同节点数的影响"""

    all_results = {}

    for num_nodes in num_nodes_list:
        print(f"\n{'='*80}")
        print(f"测试 {num_nodes} 个节点...")
        print(f"{'='*80}\n")

        node_results = {
            'Round Robin': [],
            'Shortest Queue': [],
            'Probabilistic': []
        }

        for run in range(num_runs):
            print(f"  运行 {run + 1}/{num_runs}...")

            # 生成数据
            execution_times = create_four_peak_distribution(num_tasks, seed=42 + run)
            tasks_template = generate_tasks(execution_times, arrival_rate)

            # 测试每种调度器
            schedulers = {
                'Round Robin': RoundRobinScheduler(num_nodes),
                'Shortest Queue': ShortestQueueScheduler(num_nodes, variance_weight=0.5),
                'Probabilistic': ProbabilisticScheduler(num_nodes, num_samples=1),
            }

            for name, scheduler in schedulers.items():
                # 创建任务副本
                tasks = [Task(t.task_id, t.execution_time, t.arrival_time) for t in tasks_template]

                # 运行仿真
                sim = DiscreteEventSimulator(scheduler)
                for task in tasks:
                    sim.add_arrival_event(task)
                task_metrics = sim.run()

                # 计算指标
                metrics = calculate_metrics(task_metrics)
                node_results[name].append(metrics)

        all_results[num_nodes] = node_results

    return all_results


def aggregate_results(results: Dict) -> pd.DataFrame:
    """聚合多次运行的结果"""
    aggregated = []

    for num_nodes, scheduler_results in results.items():
        for scheduler_name, runs in scheduler_results.items():
            # 计算每个指标的平均值和标准差
            avg_response_time = np.mean([r['avg_response_time'] for r in runs])
            std_response_time = np.std([r['avg_response_time'] for r in runs])
            median_response_time = np.mean([r['median_response_time'] for r in runs])
            p95_response_time = np.mean([r['p95_response_time'] for r in runs])
            p99_response_time = np.mean([r['p99_response_time'] for r in runs])

            aggregated.append({
                'num_nodes': num_nodes,
                'scheduler': scheduler_name,
                'avg_response_time': avg_response_time,
                'avg_response_time_std': std_response_time,
                'median_response_time': median_response_time,
                'p95_response_time': p95_response_time,
                'p99_response_time': p99_response_time,
            })

    return pd.DataFrame(aggregated)


def visualize_scaling_analysis(df: pd.DataFrame, filename: str = 'scaling_analysis.png'):
    """可视化扩展性分析"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = [
        ('avg_response_time', 'Average Response Time'),
        ('median_response_time', 'Median Response Time'),
        ('p95_response_time', 'P95 Response Time'),
        ('p99_response_time', 'P99 Response Time'),
    ]

    schedulers = df['scheduler'].unique()
    colors = {'Round Robin': 'blue', 'Shortest Queue': 'green', 'Probabilistic': 'red'}

    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        for scheduler in schedulers:
            data = df[df['scheduler'] == scheduler]
            x = data['num_nodes'].values
            y = data[metric].values

            ax.plot(x, y, marker='o', label=scheduler, color=colors.get(scheduler, 'black'),
                   linewidth=2, markersize=8)

            # 添加误差棒（如果有标准差）
            if metric == 'avg_response_time' and 'avg_response_time_std' in data.columns:
                yerr = data['avg_response_time_std'].values
                ax.fill_between(x, y - yerr, y + yerr, alpha=0.2, color=colors.get(scheduler, 'black'))

        ax.set_xlabel('Number of Nodes')
        ax.set_ylabel('Time (seconds)')
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nScaling analysis visualization saved to {filename}")
    plt.close()


def test_arrival_rate_impact(
    num_nodes: int,
    num_tasks: int,
    arrival_rates: List[float]
) -> Dict:
    """测试到达率对性能的影响"""

    print(f"\n{'='*80}")
    print(f"测试到达率影响（节点数: {num_nodes}，任务数: {num_tasks}）")
    print(f"{'='*80}\n")

    results = {}

    for rate in arrival_rates:
        print(f"\n测试到达率: {rate} 任务/秒...")

        # 生成数据
        execution_times = create_four_peak_distribution(num_tasks, seed=42)
        tasks_template = generate_tasks(execution_times, rate)

        rate_results = {}

        schedulers = {
            'Round Robin': RoundRobinScheduler(num_nodes),
            'Shortest Queue': ShortestQueueScheduler(num_nodes, variance_weight=0.5),
            'Probabilistic': ProbabilisticScheduler(num_nodes, num_samples=1),
        }

        for name, scheduler in schedulers.items():
            tasks = [Task(t.task_id, t.execution_time, t.arrival_time) for t in tasks_template]
            sim = DiscreteEventSimulator(scheduler)
            for task in tasks:
                sim.add_arrival_event(task)
            task_metrics = sim.run()
            metrics = calculate_metrics(task_metrics)
            rate_results[name] = metrics

        results[rate] = rate_results

    return results


def visualize_arrival_rate_analysis(results: Dict, filename: str = 'arrival_rate_analysis.png'):
    """可视化到达率分析"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = [
        ('avg_response_time', 'Average Response Time'),
        ('median_response_time', 'Median Response Time'),
        ('avg_waiting_time', 'Average Waiting Time'),
        ('p95_response_time', 'P95 Response Time'),
    ]

    arrival_rates = sorted(results.keys())
    schedulers = list(next(iter(results.values())).keys())
    colors = {'Round Robin': 'blue', 'Shortest Queue': 'green', 'Probabilistic': 'red'}

    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        for scheduler in schedulers:
            y = [results[rate][scheduler][metric] for rate in arrival_rates]
            ax.plot(arrival_rates, y, marker='o', label=scheduler,
                   color=colors.get(scheduler, 'black'), linewidth=2, markersize=8)

        ax.set_xlabel('Arrival Rate (tasks/second)')
        ax.set_ylabel('Time (seconds)')
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nArrival rate analysis visualization saved to {filename}")
    plt.close()


def print_summary_table(df: pd.DataFrame):
    """打印汇总表格"""
    print(f"\n{'='*80}")
    print("扩展性分析汇总")
    print(f"{'='*80}\n")

    for num_nodes in sorted(df['num_nodes'].unique()):
        print(f"\n节点数: {num_nodes}")
        print("-" * 80)

        subset = df[df['num_nodes'] == num_nodes]

        print(f"\n{'调度器':<20} {'平均响应时间':<15} {'中位数响应时间':<15} {'P95响应时间':<15}")
        print("-" * 80)

        for _, row in subset.iterrows():
            print(f"{row['scheduler']:<20} {row['avg_response_time']:>12.2f}    "
                  f"{row['median_response_time']:>12.2f}       "
                  f"{row['p95_response_time']:>12.2f}")


def main():
    """主函数"""
    print("开始高级分析实验...")

    # 实验1: 测试不同节点数的影响
    print("\n" + "="*80)
    print("实验1: 扩展性分析（不同节点数）")
    print("="*80)

    num_nodes_list = [2, 4, 8, 16]
    num_tasks = 1000
    arrival_rate = 0.5
    num_runs = 3

    results = run_multiple_experiments(num_nodes_list, num_tasks, arrival_rate, num_runs)
    df = aggregate_results(results)
    print_summary_table(df)
    visualize_scaling_analysis(df)

    # 实验2: 测试不同到达率的影响
    print("\n" + "="*80)
    print("实验2: 到达率影响分析")
    print("="*80)

    arrival_rates = [0.1, 0.25, 0.5, 1.0, 2.0]
    rate_results = test_arrival_rate_impact(4, 1000, arrival_rates)
    visualize_arrival_rate_analysis(rate_results)

    # 打印到达率分析结果
    print(f"\n{'='*80}")
    print("到达率影响分析汇总")
    print(f"{'='*80}\n")

    for rate in arrival_rates:
        print(f"\n到达率: {rate} 任务/秒")
        print("-" * 80)
        print(f"{'调度器':<20} {'平均响应时间':<15} {'平均等待时间':<15}")
        print("-" * 80)

        for scheduler, metrics in rate_results[rate].items():
            print(f"{scheduler:<20} {metrics['avg_response_time']:>12.2f}    "
                  f"{metrics['avg_waiting_time']:>12.2f}")

    print(f"\n{'='*80}")
    print("高级分析完成！")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
