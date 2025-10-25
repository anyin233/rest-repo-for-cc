"""
分布式调度系统建模
实现三种调度算法：Round Robin, Shortest Queue, Probabilistic
"""

import numpy as np
from typing import List, Callable, Tuple
from dataclasses import dataclass
import heapq


@dataclass
class Task:
    """任务类"""
    task_id: int
    execution_time: float  # 实际执行时间（从分布中采样）
    arrival_time: float  # 到达时间

    def __lt__(self, other):
        return self.task_id < other.task_id


class Queue:
    """节点队列类"""
    def __init__(self, node_id: int):
        self.node_id = node_id
        self.tasks: List[Task] = []
        self.total_time = 0.0  # 当前队列总时间

    def add_task(self, task: Task):
        """添加任务到队列"""
        self.tasks.append(task)
        self.total_time += task.execution_time

    def get_expected_time(self) -> float:
        """获取队列总时间期望"""
        return self.total_time

    def get_variance(self) -> float:
        """获取队列总时间方差
        简化：假设任务执行时间独立，总方差为各任务方差之和
        这里使用执行时间的平方和作为方差的近似
        """
        if not self.tasks:
            return 0.0
        # 简化的方差估计
        mean = self.total_time / len(self.tasks) if self.tasks else 0
        variance = sum((task.execution_time - mean) ** 2 for task in self.tasks)
        return variance

    def sample_total_time(self) -> float:
        """采样队列总时间
        对于已经确定的任务，直接返回总时间
        """
        return self.total_time

    def is_empty(self) -> bool:
        return len(self.tasks) == 0

    def peek_next_task(self) -> Task:
        """查看下一个要执行的任务"""
        return self.tasks[0] if self.tasks else None

    def complete_task(self) -> Task:
        """完成并移除队列头部任务"""
        if self.tasks:
            task = self.tasks.pop(0)
            self.total_time -= task.execution_time
            return task
        return None


class Scheduler:
    """调度器基类"""
    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes
        self.queues = [Queue(i) for i in range(num_nodes)]

    def schedule(self, task: Task) -> int:
        """调度任务到某个节点，返回节点ID"""
        raise NotImplementedError

    def add_task(self, task: Task):
        """添加任务"""
        node_id = self.schedule(task)
        self.queues[node_id].add_task(task)
        return node_id

    def get_total_queue_time(self) -> float:
        """获取所有队列的总时间"""
        return sum(q.total_time for q in self.queues)


class RoundRobinScheduler(Scheduler):
    """Round Robin调度器"""
    def __init__(self, num_nodes: int):
        super().__init__(num_nodes)
        self.current_node = 0

    def schedule(self, task: Task) -> int:
        """轮询调度"""
        node_id = self.current_node
        self.current_node = (self.current_node + 1) % self.num_nodes
        return node_id


class ShortestQueueScheduler(Scheduler):
    """Shortest Queue调度器（基于期望和方差）"""
    def __init__(self, num_nodes: int, variance_weight: float = 0.5):
        super().__init__(num_nodes)
        self.variance_weight = variance_weight

    def schedule(self, task: Task) -> int:
        """选择最短队列"""
        min_score = float('inf')
        best_node = 0

        for i, queue in enumerate(self.queues):
            expected = queue.get_expected_time()
            variance = queue.get_variance()
            # 综合考虑期望和方差
            score = expected + self.variance_weight * np.sqrt(variance)

            if score < min_score:
                min_score = score
                best_node = i

        return best_node


class ProbabilisticScheduler(Scheduler):
    """Probabilistic调度器（基于采样）"""
    def __init__(self, num_nodes: int, num_samples: int = 1):
        super().__init__(num_nodes)
        self.num_samples = num_samples

    def schedule(self, task: Task) -> int:
        """基于采样的概率调度"""
        # 对每个队列进行采样
        samples = []
        for i, queue in enumerate(self.queues):
            # 进行多次采样，计算平均值
            queue_samples = [queue.sample_total_time() for _ in range(self.num_samples)]
            avg_sample = np.mean(queue_samples)
            samples.append((avg_sample, i))

        # 找到采样值最小的队列
        min_sample, best_node = min(samples)
        return best_node


class DiscreteEventSimulator:
    """离散事件仿真器"""
    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler
        self.current_time = 0.0
        self.events = []  # 优先队列：(time, event_type, data)
        self.completed_tasks = []

    def add_arrival_event(self, task: Task):
        """添加任务到达事件"""
        heapq.heappush(self.events, (task.arrival_time, 'arrival', task))

    def run(self) -> List[Tuple[int, float, float, float]]:
        """运行仿真
        返回：[(task_id, arrival_time, start_time, completion_time), ...]
        """
        # 跟踪每个节点的当前时间（何时空闲）
        node_free_time = [0.0] * self.scheduler.num_nodes
        task_metrics = []

        while self.events:
            event_time, event_type, data = heapq.heappop(self.events)
            self.current_time = event_time

            if event_type == 'arrival':
                task = data
                # 调度任务
                node_id = self.scheduler.add_task(task)

                # 计算任务开始时间（max of arrival time and node free time）
                start_time = max(task.arrival_time, node_free_time[node_id])
                completion_time = start_time + task.execution_time

                # 更新节点空闲时间
                node_free_time[node_id] = completion_time

                # 记录任务指标
                task_metrics.append((
                    task.task_id,
                    task.arrival_time,
                    start_time,
                    completion_time
                ))

        return task_metrics


def calculate_metrics(task_metrics: List[Tuple[int, float, float, float]]) -> dict:
    """计算性能指标"""
    waiting_times = []
    response_times = []

    for task_id, arrival, start, completion in task_metrics:
        waiting_time = start - arrival
        response_time = completion - arrival
        waiting_times.append(waiting_time)
        response_times.append(response_time)

    return {
        'avg_waiting_time': np.mean(waiting_times),
        'avg_response_time': np.mean(response_times),
        'max_waiting_time': np.max(waiting_times),
        'max_response_time': np.max(response_times),
        'std_waiting_time': np.std(waiting_times),
        'std_response_time': np.std(response_times),
        'median_waiting_time': np.median(waiting_times),
        'median_response_time': np.median(response_times),
        'p95_waiting_time': np.percentile(waiting_times, 95),
        'p95_response_time': np.percentile(response_times, 95),
        'p99_waiting_time': np.percentile(waiting_times, 99),
        'p99_response_time': np.percentile(response_times, 99),
    }
