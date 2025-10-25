"""
演示队列动态更新机制
详细展示任务到达和完成时队列状态的变化
"""

import numpy as np
from realistic_scheduler import (
    Task, TaskDistribution, ShortestQueueScheduler, ProbabilisticScheduler,
    RealisticSimulator
)
import matplotlib.pyplot as plt
from typing import List, Dict


class MonitoredSimulator(RealisticSimulator):
    """带监控功能的仿真器"""

    def __init__(self, scheduler):
        super().__init__(scheduler)
        self.queue_history = []  # 记录队列状态历史

    def record_queue_state(self, event_time: float, event_type: str, task_id: int = -1):
        """记录队列状态"""
        state = {
            'time': event_time,
            'event_type': event_type,
            'task_id': task_id,
            'queues': []
        }

        for i, queue in enumerate(self.scheduler.queues):
            queue_info = {
                'node_id': i,
                'num_waiting': len(queue.tasks),
                'has_running': queue.current_task is not None,
                'mean': queue.distribution.mean,
                'variance': queue.distribution.variance,
                'quantiles': queue.distribution.quantiles.copy()
            }
            state['queues'].append(queue_info)

        self.queue_history.append(state)

    def run(self) -> List[Task]:
        """运行仿真并记录状态"""
        self.record_queue_state(0.0, 'initial')

        while self.events:
            event_time, event_type, data = self.events[0]
            self.events.pop(0)

            if event_type == 'arrival':
                task = data
                self.scheduler.on_task_arrival(task, event_time)
                self.record_queue_state(event_time, 'arrival', task.task_id)

                node_id = task.assigned_node
                started_task = self.scheduler.start_task_if_idle(node_id, event_time)

                if started_task:
                    self.events.append((started_task.completion_time, 'completion', started_task))
                    self.events.sort(key=lambda x: x[0])
                    self.record_queue_state(event_time, 'start', started_task.task_id)

            elif event_type == 'completion':
                completed_task = data
                node_id = completed_task.assigned_node

                self.scheduler.on_task_completion(node_id, event_time)
                self.completed_tasks.append(completed_task)
                self.record_queue_state(event_time, 'completion', completed_task.task_id)

                next_task = self.scheduler.start_task_if_idle(node_id, event_time)
                if next_task:
                    self.events.append((next_task.completion_time, 'completion', next_task))
                    self.events.sort(key=lambda x: x[0])
                    self.record_queue_state(event_time, 'start', next_task.task_id)

        return self.completed_tasks


def demo_shortest_queue_updates():
    """演示Shortest Queue调度器的队列更新"""
    print("\n" + "="*80)
    print("演示：Shortest Queue 调度器的动态队列更新")
    print("="*80 + "\n")

    np.random.seed(42)
    num_nodes = 3
    num_tasks = 10

    # 创建任务分布（简化为单峰）
    task_dist = TaskDistribution.from_normal(mean=100, std=15)

    # 生成任务
    tasks = []
    for i in range(num_tasks):
        task = Task(
            task_id=i,
            distribution=task_dist,
            actual_execution_time=task_dist.sample(),
            arrival_time=i * 20.0  # 每20秒到达一个任务
        )
        tasks.append(task)

    # 创建调度器和仿真器
    scheduler = ShortestQueueScheduler(num_nodes, variance_weight=0.5)
    sim = MonitoredSimulator(scheduler)

    # 添加任务
    for task in tasks:
        sim.add_task_arrival(task)

    # 运行仿真
    completed = sim.run()

    print(f"完成 {len(completed)} 个任务\n")
    print("关键时刻的队列状态变化:\n")

    # 显示前20个事件
    for idx, state in enumerate(sim.queue_history[:20]):
        print(f"时间 {state['time']:.2f}s - 事件: {state['event_type']} (任务 {state['task_id']})")
        print("-" * 80)

        for q in state['queues']:
            status = "运行中" if q['has_running'] else "空闲"
            print(f"  节点 {q['node_id']}: 等待任务={q['num_waiting']}, 状态={status}, "
                  f"期望时间={q['mean']:.2f}s, 方差={q['variance']:.2f}")
        print()

    print("\n" + "="*80)
    print("观察要点:")
    print("="*80)
    print("1. 任务到达时(arrival)，被分配到期望时间最短的队列")
    print("2. 队列的期望时间和方差随着任务添加而增加")
    print("3. 任务完成时(completion)，队列的期望时间和方差相应减少")
    print("4. 最终所有队列的期望时间和方差都回到接近零")


def demo_probabilistic_updates():
    """演示Probabilistic调度器的分位数更新"""
    print("\n" + "="*80)
    print("演示：Probabilistic 调度器的分位数动态更新")
    print("="*80 + "\n")

    np.random.seed(42)
    num_nodes = 3
    num_tasks = 10

    # 创建任务分布
    task_dist = TaskDistribution.from_normal(mean=100, std=15)

    # 生成任务
    tasks = []
    for i in range(num_tasks):
        task = Task(
            task_id=i,
            distribution=task_dist,
            actual_execution_time=task_dist.sample(),
            arrival_time=i * 20.0
        )
        tasks.append(task)

    # 创建调度器和仿真器
    scheduler = ProbabilisticScheduler(num_nodes)
    sim = MonitoredSimulator(scheduler)

    # 添加任务
    for task in tasks:
        sim.add_task_arrival(task)

    # 运行仿真
    completed = sim.run()

    print(f"完成 {len(completed)} 个任务\n")
    print("关键时刻的队列分位数变化:\n")

    # 显示前20个事件
    for idx, state in enumerate(sim.queue_history[:20]):
        print(f"时间 {state['time']:.2f}s - 事件: {state['event_type']} (任务 {state['task_id']})")
        print("-" * 80)

        for q in state['queues']:
            status = "运行中" if q['has_running'] else "空闲"
            # 显示P25, P50, P75, P95
            p25 = q['quantiles'][2]
            p50 = q['quantiles'][3]
            p75 = q['quantiles'][4]
            p95 = q['quantiles'][6]

            print(f"  节点 {q['node_id']}: 等待={q['num_waiting']}, {status}, "
                  f"P25={p25:.1f}s, P50={p50:.1f}s, P75={p75:.1f}s, P95={p95:.1f}s")
        print()

    print("\n" + "="*80)
    print("观察要点:")
    print("="*80)
    print("1. 任务到达时(arrival)，队列的所有分位数都增加")
    print("2. 调度器基于分位数采样选择队列（采样值最小的队列）")
    print("3. 任务完成时(completion)，从队列分布中减去任务的分布")
    print("4. 分位数的加减法使得我们能够跟踪整个队列时间的概率分布")
    print("5. 最终所有分位数都回到接近零，表示队列已清空")


def visualize_queue_dynamics(scheduler_name: str, num_nodes: int = 3, num_tasks: int = 30):
    """可视化队列动态变化"""
    print(f"\n生成 {scheduler_name} 调度器的队列动态可视化...")

    np.random.seed(42)

    # 创建任务分布
    task_dist = TaskDistribution.from_normal(mean=100, std=15)

    # 生成任务
    tasks = []
    for i in range(num_tasks):
        task = Task(
            task_id=i,
            distribution=task_dist,
            actual_execution_time=task_dist.sample(),
            arrival_time=i * 15.0
        )
        tasks.append(task)

    # 创建调度器
    if scheduler_name == "Shortest Queue":
        scheduler = ShortestQueueScheduler(num_nodes, variance_weight=0.5)
    else:
        scheduler = ProbabilisticScheduler(num_nodes)

    sim = MonitoredSimulator(scheduler)

    for task in tasks:
        sim.add_task_arrival(task)

    completed = sim.run()

    # 提取时间序列数据
    times = [s['time'] for s in sim.queue_history]
    queue_means = {i: [] for i in range(num_nodes)}

    for state in sim.queue_history:
        for q in state['queues']:
            queue_means[q['node_id']].append(q['mean'])

    # 绘图
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # 子图1: 队列期望时间变化
    ax1 = axes[0]
    for i in range(num_nodes):
        ax1.plot(times, queue_means[i], label=f'Queue {i}', linewidth=2, marker='o', markersize=3)

    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Expected Queue Time (seconds)')
    ax1.set_title(f'{scheduler_name}: Queue Expected Time Over Time')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 子图2: 任务分配分布
    ax2 = axes[1]
    assignments = [0] * num_nodes
    for task in completed:
        assignments[task.assigned_node] += 1

    ax2.bar(range(num_nodes), assignments, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Node ID')
    ax2.set_ylabel('Number of Tasks')
    ax2.set_title(f'{scheduler_name}: Task Distribution Across Nodes')
    ax2.set_xticks(range(num_nodes))
    ax2.grid(alpha=0.3, axis='y')

    # 添加数值标签
    for i, v in enumerate(assignments):
        ax2.text(i, v, str(v), ha='center', va='bottom')

    plt.tight_layout()
    filename = f'queue_dynamics_{scheduler_name.replace(" ", "_").lower()}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  已保存到 {filename}")
    plt.close()


def main():
    """主函数"""
    # 演示Shortest Queue更新
    demo_shortest_queue_updates()

    # 演示Probabilistic更新
    demo_probabilistic_updates()

    # 可视化队列动态
    print("\n" + "="*80)
    print("生成队列动态可视化图表")
    print("="*80)

    visualize_queue_dynamics("Shortest Queue", num_nodes=3, num_tasks=30)
    visualize_queue_dynamics("Probabilistic", num_nodes=3, num_tasks=30)

    print("\n" + "="*80)
    print("演示完成!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
