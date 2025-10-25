"""
真实分布式调度系统建模
每个节点必须处理完当前任务才能处理下一个
任务完成时动态更新队列统计信息
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
import heapq
from scipy import stats


@dataclass
class TaskDistribution:
    """任务执行时间分布（用分位数表示）"""
    quantiles: np.ndarray  # 分位数数组，如[P5, P10, P25, P50, P75, P90, P95]
    quantile_levels: np.ndarray  # 对应的概率水平，如[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    mean: float  # 期望
    variance: float  # 方差

    @classmethod
    def from_normal(cls, mean: float, std: float):
        """从正态分布创建任务分布"""
        quantile_levels = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
        quantiles = stats.norm.ppf(quantile_levels, loc=mean, scale=std)
        return cls(
            quantiles=quantiles,
            quantile_levels=quantile_levels,
            mean=mean,
            variance=std ** 2
        )

    def sample(self) -> float:
        """从分布中采样一个执行时间"""
        # 使用线性插值在分位数之间采样
        p = np.random.random()
        value = np.interp(p, self.quantile_levels, self.quantiles)
        return max(value, 1.0)  # 确保执行时间为正

    def add(self, other: 'TaskDistribution') -> 'TaskDistribution':
        """分布加法（用于队列添加任务）"""
        # 近似：独立随机变量的和的分位数 ≈ 分位数之和
        new_quantiles = self.quantiles + other.quantiles
        new_mean = self.mean + other.mean
        new_variance = self.variance + other.variance
        return TaskDistribution(
            quantiles=new_quantiles,
            quantile_levels=self.quantile_levels.copy(),
            mean=new_mean,
            variance=new_variance
        )

    def subtract(self, other: 'TaskDistribution') -> 'TaskDistribution':
        """分布减法（用于队列移除任务）"""
        # 近似：X - Y 的分位数
        # Q_{X-Y}(p) ≈ Q_X(p) - Q_Y(1-p)
        # 但为了简化，我们直接减去（保持对称性）
        new_quantiles = self.quantiles - other.quantiles
        new_mean = self.mean - other.mean
        new_variance = max(self.variance - other.variance, 0.0)  # 方差不能为负

        return TaskDistribution(
            quantiles=new_quantiles,
            quantile_levels=self.quantile_levels.copy(),
            mean=max(new_mean, 0.0),
            variance=new_variance
        )

    @classmethod
    def empty(cls):
        """创建空分布（零分布）"""
        quantile_levels = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
        quantiles = np.zeros_like(quantile_levels)
        return cls(
            quantiles=quantiles,
            quantile_levels=quantile_levels,
            mean=0.0,
            variance=0.0
        )


@dataclass
class Task:
    """任务类"""
    task_id: int
    distribution: TaskDistribution  # 执行时间分布
    actual_execution_time: float  # 实际执行时间（采样值）
    arrival_time: float  # 到达时间
    assigned_node: int = -1  # 分配的节点ID
    start_time: float = -1  # 开始执行时间
    completion_time: float = -1  # 完成时间


class QueueState:
    """队列状态类"""
    def __init__(self, node_id: int):
        self.node_id = node_id
        self.tasks: List[Task] = []  # 等待中的任务
        self.current_task: Task = None  # 正在执行的任务
        self.distribution = TaskDistribution.empty()  # 队列总时间分布
        self.expected_completion_time = 0.0  # 预期完成时间

    def add_task(self, task: Task, current_time: float):
        """添加任务到队列"""
        self.tasks.append(task)
        # 更新分布（加上新任务的分布）
        self.distribution = self.distribution.add(task.distribution)

    def start_next_task(self, current_time: float) -> Task:
        """开始执行下一个任务"""
        if self.tasks:
            self.current_task = self.tasks.pop(0)
            self.current_task.start_time = current_time
            self.current_task.completion_time = current_time + self.current_task.actual_execution_time
            return self.current_task
        return None

    def complete_current_task(self) -> Task:
        """完成当前任务"""
        if self.current_task:
            completed_task = self.current_task
            # 更新分布（减去完成任务的分布）
            self.distribution = self.distribution.subtract(completed_task.distribution)
            self.current_task = None
            return completed_task
        return None

    def get_next_free_time(self, current_time: float) -> float:
        """获取节点下次空闲时间"""
        if self.current_task:
            return self.current_task.completion_time
        return current_time

    def is_idle(self) -> bool:
        """节点是否空闲"""
        return self.current_task is None


class Scheduler:
    """调度器基类"""
    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes
        self.queues = [QueueState(i) for i in range(num_nodes)]

    def schedule_task(self, task: Task, current_time: float) -> int:
        """调度任务到某个节点，返回节点ID"""
        raise NotImplementedError

    def on_task_arrival(self, task: Task, current_time: float):
        """任务到达时的处理"""
        node_id = self.schedule_task(task, current_time)
        task.assigned_node = node_id
        self.queues[node_id].add_task(task, current_time)

    def on_task_completion(self, node_id: int, current_time: float) -> Task:
        """任务完成时的处理"""
        return self.queues[node_id].complete_current_task()

    def start_task_if_idle(self, node_id: int, current_time: float) -> Task:
        """如果节点空闲，开始执行下一个任务"""
        if self.queues[node_id].is_idle():
            return self.queues[node_id].start_next_task(current_time)
        return None


class RoundRobinScheduler(Scheduler):
    """Round Robin调度器"""
    def __init__(self, num_nodes: int):
        super().__init__(num_nodes)
        self.current_node = 0

    def schedule_task(self, task: Task, current_time: float) -> int:
        """轮询调度"""
        node_id = self.current_node
        self.current_node = (self.current_node + 1) % self.num_nodes
        return node_id


class ShortestQueueScheduler(Scheduler):
    """Shortest Queue调度器（基于期望和方差）"""
    def __init__(self, num_nodes: int, variance_weight: float = 0.5):
        super().__init__(num_nodes)
        self.variance_weight = variance_weight

    def schedule_task(self, task: Task, current_time: float) -> int:
        """选择期望时间最短的队列"""
        min_score = float('inf')
        best_node = 0

        for i, queue in enumerate(self.queues):
            # 综合考虑期望和方差
            dist = queue.distribution
            score = dist.mean + self.variance_weight * np.sqrt(dist.variance)

            if score < min_score:
                min_score = score
                best_node = i

        return best_node


class ProbabilisticScheduler(Scheduler):
    """Probabilistic调度器（基于分位数分布采样）"""
    def __init__(self, num_nodes: int):
        super().__init__(num_nodes)

    def schedule_task(self, task: Task, current_time: float) -> int:
        """基于采样选择队列"""
        # 对每个队列的分布进行一次采样
        samples = []
        for i, queue in enumerate(self.queues):
            # 从队列的分位数分布中采样
            sample = self._sample_from_quantiles(queue.distribution)
            samples.append((sample, i))

        # 选择采样值最小的队列
        _, best_node = min(samples)
        return best_node

    def _sample_from_quantiles(self, dist: TaskDistribution) -> float:
        """从分位数表示的分布中采样"""
        # 随机选择一个概率水平
        p = np.random.random()
        # 在分位数之间线性插值
        value = np.interp(p, dist.quantile_levels, dist.quantiles)
        return max(value, 0.0)


class RealisticSimulator:
    """真实的离散事件仿真器"""
    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler
        self.events = []  # 优先队列：(time, event_type, data)
        self.completed_tasks = []

    def add_task_arrival(self, task: Task):
        """添加任务到达事件"""
        heapq.heappush(self.events, (task.arrival_time, 'arrival', task))

    def run(self) -> List[Task]:
        """运行仿真"""
        while self.events:
            event_time, event_type, data = heapq.heappop(self.events)

            if event_type == 'arrival':
                task = data
                # 任务到达：调度并加入队列
                self.scheduler.on_task_arrival(task, event_time)

                # 尝试在分配的节点上开始执行
                node_id = task.assigned_node
                started_task = self.scheduler.start_task_if_idle(node_id, event_time)

                if started_task:
                    # 如果任务开始执行，添加完成事件
                    heapq.heappush(
                        self.events,
                        (started_task.completion_time, 'completion', started_task)
                    )

            elif event_type == 'completion':
                completed_task = data
                node_id = completed_task.assigned_node

                # 任务完成：从队列中移除
                self.scheduler.on_task_completion(node_id, event_time)
                self.completed_tasks.append(completed_task)

                # 尝试开始下一个任务
                next_task = self.scheduler.start_task_if_idle(node_id, event_time)
                if next_task:
                    # 添加下一个任务的完成事件
                    heapq.heappush(
                        self.events,
                        (next_task.completion_time, 'completion', next_task)
                    )

        return self.completed_tasks


def calculate_metrics(tasks: List[Task]) -> dict:
    """计算性能指标"""
    waiting_times = []
    response_times = []

    for task in tasks:
        waiting_time = task.start_time - task.arrival_time
        response_time = task.completion_time - task.arrival_time
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
