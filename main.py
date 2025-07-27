#!/usr/bin/env python3
"""
Multi-Agent Intelligent Systems Project - Student Assignment
Course: Artificial Intelligence - Intelligent Agents and Multi-Agent Systems
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Dict, List, Tuple, Set, Optional, Any
from enum import Enum
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import heapq
import random
import time
import json
from pathlib import Path
from matplotlib.animation import FuncAnimation
import pygame
import sys


# ============================================================================
# CORE ENVIRONMENT IMPLEMENTATION (PROVIDED - DO NOT MODIFY)
# ============================================================================

class CellType(Enum):
    """Defines the types of cells in the gridworld"""
    EMPTY = 0
    WALL = 1
    GOAL = 2
    RESOURCE = 3
    HAZARD = 4


class Direction(Enum):
    """Defines possible movement directions"""
    NORTH = (0, -1)
    SOUTH = (0, 1)
    EAST = (1, 0)
    WEST = (-1, 0)


class Action(Enum):
    """Defines possible actions an agent can take"""
    MOVE_NORTH = "move_north"
    MOVE_SOUTH = "move_south"
    MOVE_EAST = "move_east"
    MOVE_WEST = "move_west"
    PICKUP = "pickup"
    DROP = "drop"
    WAIT = "wait"
    COMMUNICATE = "communicate"


@dataclass
class Position:
    """Represents a position in the gridworld"""
    x: int
    y: int

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __lt__(self, other):
        """Define ordering for priority queue operations"""
        if self.x != other.x:
            return self.x < other.x
        return self.y < other.y

    def __add__(self, direction: Direction):
        """Move position in given direction"""
        dx, dy = direction.value
        return Position(self.x + dx, self.y + dy)


@dataclass
class Message:
    """Represents communication between agents"""
    sender_id: int
    recipient_id: int  # -1 for broadcast
    content: str
    timestamp: int


@dataclass
class Perception:
    """Represents what an agent can perceive from its current position"""
    position: Position
    visible_cells: Dict[Position, CellType]
    visible_agents: Dict[int, Position]  # agent_id -> position
    energy_level: float
    has_resource: bool
    messages: List[Message]


# Environment implementation provided - students use this as-is
class GridWorld:
    def __init__(self, width: int, height: int, max_agents: int = 4):
        self.width = width
        self.height = height
        self.max_agents = max_agents
        self.time_step = 0
        self.grid = np.full((height, width), CellType.EMPTY)

        self.agents: Dict[int, Agent] = {}
        self.agent_positions: Dict[int, Position] = {}
        self.agent_energy: Dict[int, float] = {}
        self.agent_resources: Dict[int, int] = {}

        self.message_queue: List[Message] = []
        self.communication_range = 3

        self.goals: Set[Position] = set()
        self.resources: Set[Position] = set()
        self.hazards: Set[Position] = set()
        self.task_completion_times: List[int] = []

        self.collision_count = 0
        self.total_energy_consumed = 0
        self.logs = []

        self.initial_resource_count = 0
        self.total_navigable_cells = 0

    # -------------------
    # Environment setup
    # -------------------
    def add_walls(self, wall_positions: List[Position]):
        for pos in wall_positions:
            if self.is_valid_position(pos):
                self.grid[pos.y, pos.x] = CellType.WALL
        self.total_navigable_cells = np.sum(self.grid != CellType.WALL)

    def add_resources(self, resource_positions: List[Position]):
        for pos in resource_positions:
            if self.is_valid_position(pos):
                self.grid[pos.y, pos.x] = CellType.RESOURCE
                self.resources.add(pos)
        self.initial_resource_count = len(self.resources)

    def add_goals(self, goal_positions: List[Position]):
        for pos in goal_positions:
            if self.is_valid_position(pos):
                self.grid[pos.y, pos.x] = CellType.GOAL
                self.goals.add(pos)

    def add_hazards(self, hazard_positions: List[Position]):
        for pos in hazard_positions:
            if self.is_valid_position(pos):
                self.grid[pos.y, pos.x] = CellType.HAZARD
                self.hazards.add(pos)

    def add_agent(self, agent: "Agent", position: Position) -> bool:
        if len(self.agents) < self.max_agents and self.is_position_free(position):
            agent_id = len(self.agents)
            self.agents[agent_id] = agent
            agent.agent_id = agent_id
            self.agent_positions[agent_id] = position
            self.agent_energy[agent_id] = 100.0
            self.agent_resources[agent_id] = 0
            return True
        return False

    #------------------- Helper functions -------------------
    def is_valid_position(self, position: Position) -> bool:
        return 0 <= position.x < self.width and 0 <= position.y < self.height

    def is_position_free(self, position: Position) -> bool:
        if not self.is_valid_position(position):
            return False
        if self.grid[position.y, position.x] == CellType.WALL:
            return False
        for pos in self.agent_positions.values():
            if pos == position:
                return False
        return True

    def get_perception(self, agent_id: int, perception_range: int = 2) -> Perception:
        position = self.agent_positions[agent_id]
        visible_cells = {}
        for dy in range(-perception_range, perception_range + 1):
            for dx in range(-perception_range, perception_range + 1):
                x, y = position.x + dx, position.y + dy
                if 0 <= x < self.width and 0 <= y < self.height:
                    visible_cells[Position(x, y)] = self.grid[y, x]
        visible_agents = {
            aid: pos for aid, pos in self.agent_positions.items()
            if aid != agent_id and abs(pos.x - position.x) <= perception_range and abs(
                pos.y - position.y) <= perception_range
        }
        messages = [msg for msg in self.message_queue if msg.recipient_id in (agent_id, -1)]
        return Perception(
            position=position,
            visible_cells=visible_cells,
            visible_agents=visible_agents,
            energy_level=self.agent_energy[agent_id],
            has_resource=self.agent_resources[agent_id] > 0,
            messages=messages
        )


    # -------------------Core simulation step-------------------
    def execute_action(self, agent_id: int, action: Action, message_content: str = "") -> bool:
        position = self.agent_positions[agent_id]
        success = False

        if self.agent_energy[agent_id] <= 0:
            return False

        # Movement
        if action in {Action.MOVE_NORTH, Action.MOVE_SOUTH, Action.MOVE_EAST, Action.MOVE_WEST}:
            direction = {
                Action.MOVE_NORTH: Direction.NORTH,
                Action.MOVE_SOUTH: Direction.SOUTH,
                Action.MOVE_EAST: Direction.EAST,
                Action.MOVE_WEST: Direction.WEST
            }[action]
            new_pos = position + direction
            if self.is_position_free(new_pos):
                self.agent_positions[agent_id] = new_pos
                success = True
            else:
                self.collision_count += 1

        # PICKUP
        elif action == Action.PICKUP:
            if self.grid[position.y, position.x] == CellType.RESOURCE:
                self.grid[position.y, position.x] = CellType.EMPTY
                self.resources.remove(position)
                self.agent_resources[agent_id] += 1
                success = True

        # DROP
        elif action == Action.DROP:
            if self.grid[position.y, position.x] == CellType.GOAL and self.agent_resources[agent_id] > 0:
                self.agent_resources[agent_id] -= 1
                self.task_completion_times.append(self.time_step)
                success = True

        # COMMUNICATE
        elif action == Action.COMMUNICATE:
            self.message_queue.append(Message(agent_id, -1, message_content, self.time_step))
            success = True

        # WAIT
        elif action == Action.WAIT:
            success = True

        # Energy updates
        self.energy_costs = {
            Action.MOVE_NORTH: 1.0,
            Action.MOVE_SOUTH: 1.0,
            Action.MOVE_EAST: 1.0,
            Action.MOVE_WEST: 1.0,
            Action.PICKUP: 0.5,
            Action.DROP: 0.5,
            Action.WAIT: 0.1,
            Action.COMMUNICATE: 0.5
        }
        energy_cost = self.energy_costs.get(action, 1.0)
        if self.grid[self.agent_positions[agent_id].y, self.agent_positions[agent_id].x] == CellType.HAZARD:
            energy_cost += 4.0  # Additional cost for standing on hazard

        self.agent_energy[agent_id] -= energy_cost
        self.total_energy_consumed += energy_cost
        if self.agent_energy[agent_id] < 0:
            self.agent_energy[agent_id] = 0
        return success

    def step(self) -> Dict[int, bool]:
        self.time_step += 1
        actions_taken = {}
        self.message_queue.clear()
        for agent_id, agent in self.agents.items():
            perception = self.get_perception(agent_id)
            if self.agent_energy[agent_id] <= 0:
                actions_taken[agent_id] = False
                self.logs.append({
                    "time_step": self.time_step,
                    "agent_id": agent_id,
                    "position": {"x": perception.position.x, "y": perception.position.y},
                    "energy": self.agent_energy[agent_id],
                    "has_resource": self.agent_resources[agent_id] > 0,
                    "action": "NO_ACTION",
                    "reason": "Agent out of energy"
                })
                continue

            action, reason = agent.decide_action(perception)
            result = self.execute_action(agent_id, action)

            self.logs.append({
                "time_step": self.time_step,
                "agent_id": agent_id,
                "position": {"x": perception.position.x, "y": perception.position.y},
                "energy": self.agent_energy[agent_id],
                "has_resource": self.agent_resources[agent_id] > 0,
                "action": action.name,
                "reason": reason
            })

            actions_taken[agent_id] = result
        return actions_taken

    def get_performance_metrics(self) -> Dict:
        if not self.agents:
            return {}

        total_resources_collected = len(self.task_completion_times)
        average_energy_remaining = sum(self.agent_energy.values()) / len(self.agent_energy)

        # Success Rate: Percentage of initial resources that were collected.
        success_rate = (
                                   total_resources_collected / self.initial_resource_count) * 100 if self.initial_resource_count > 0 else 0

        # Average Task Completion Time: Average time steps to deliver a resource.
        avg_task_completion_time = np.mean(self.task_completion_times) if self.task_completion_times else 0

        # Exploration Coverage: Percentage of the navigable area visited by agents.
        all_visited = set()
        for agent in self.agents.values():
            if hasattr(agent, 'visited_positions'):
                all_visited.update(agent.visited_positions)
        exploration_coverage = (
                                           len(all_visited) / self.total_navigable_cells) * 100 if self.total_navigable_cells > 0 else 0

        # Energy Efficiency: How many resources were collected per unit of energy.
        energy_efficiency = total_resources_collected / self.total_energy_consumed if self.total_energy_consumed > 0 else 0

        # Overall Efficiency Score: A composite score balancing success against costs.
        # This formula rewards success and penalizes time, energy, and collisions.
        cost = self.time_step + (self.total_energy_consumed * 0.1) + (self.collision_count * 5)
        efficiency_score = (total_resources_collected * 100) / (cost + 1)  # +1 to avoid division by zero

        return {
            "Total Resources Collected": total_resources_collected,
            "Success Rate (%)": success_rate,
            "Average Energy Remaining": average_energy_remaining,
            "Total Steps": self.time_step,
            "Collision Count": self.collision_count,
            "Exploration Coverage (%)": exploration_coverage,
            "Average Task Completion Time": avg_task_completion_time,
            "Energy Efficiency (Res/Unit)": energy_efficiency,
            "Overall Efficiency Score": efficiency_score,
        }

# ============================================================================
# AGENT IMPLEMENTATIONS - YOUR ASSIGNMENT BEGINS HERE
# ============================================================================

class Agent(ABC):
    """Abstract base class for all agent implementations"""
    def __init__(self, name: str):
        self.name = name
        self.agent_id: int = -1
        self.action_history: List[Action] = []
        self.total_rewards = 0

    @abstractmethod
    def decide_action(self, perception: Perception) -> Tuple[Action, str]:
        """Decide the next action based on current perception"""
        pass

    def reset(self):
        """Reset agent state for new episode"""
        self.action_history.clear()
        self.total_rewards = 0


class SimpleReflexAgent(Agent):

    def __init__(self, name: str):
        super().__init__(name)
        self.rule_counts = defaultdict(int)

    def decide_action(self, perception: Perception) -> Tuple[Action, str]:
        pos = perception.position
        cells = perception.visible_cells

        # Rule 1: Hazard Avoidance
        if cells.get(pos) == CellType.HAZARD:
            for d in Direction:
                new_pos = pos + d
                if cells.get(new_pos) and cells[new_pos] not in (CellType.HAZARD, CellType.WALL):
                    self.rule_counts['avoid_hazard'] += 1
                    return self._direction_to_action(d), f"Avoiding hazard at {pos}"

        # Rule 2: Resource Collection
        if cells.get(pos) == CellType.RESOURCE and not perception.has_resource:
            self.rule_counts['pickup'] += 1
            return Action.PICKUP, f"Picking up resource at {pos}"

        # Rule 3: Goal Seeking (if carrying resource)
        if perception.has_resource:
            if cells.get(pos) == CellType.GOAL:
                self.rule_counts['drop_at_goal'] += 1
                return Action.DROP, f"Dropping resource at goal {pos}"
            goals = [p for p, t in cells.items() if t == CellType.GOAL]
            if goals:
                d = self._get_direction_toward(pos, goals[0], perception)
                if d:
                    self.rule_counts['to_goal'] += 1
                    return self._direction_to_action(d), f"Moving toward goal at {goals[0]}"

        # Rule 4: Resource Pursuit (if not carrying)
        if not perception.has_resource:
            resources = [p for p, t in cells.items() if t == CellType.RESOURCE]
            if resources:
                d = self._get_direction_toward(pos, resources[0], perception)
                if d:
                    self.rule_counts['to_resource'] += 1
                    return self._direction_to_action(d), f"Moving toward resource at {resources[0]}"

        # Rule 5: Random Exploration
        self.rule_counts['random'] += 1
        return self._random_valid_move(cells, pos), f"Exploring randomly from {pos}"

    def _direction_to_action(self, direction: Direction) -> Action:
        mapping = {
            Direction.NORTH: Action.MOVE_NORTH,
            Direction.SOUTH: Action.MOVE_SOUTH,
            Direction.EAST: Action.MOVE_EAST,
            Direction.WEST: Action.MOVE_WEST,
        }
        return mapping[direction]

    def _get_direction_toward(self, from_pos: Position, to_pos: Position, perception) -> Optional[Direction]:
        dx = to_pos.x - from_pos.x
        dy = to_pos.y - from_pos.y
        if dx == 0 and dy == 0:
            return None

        preferred_directions = []
        if abs(dx) >= abs(dy):
            preferred_directions.append(Direction.EAST if dx > 0 else Direction.WEST)
            if dy != 0:
                preferred_directions.append(Direction.SOUTH if dy > 0 else Direction.NORTH)
        else:
            preferred_directions.append(Direction.SOUTH if dy > 0 else Direction.NORTH)
            if dx != 0:
                preferred_directions.append(Direction.EAST if dx > 0 else Direction.WEST)

        # Add other directions as fallback
        for d in [Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST]:
            if d not in preferred_directions:
                preferred_directions.append(d)

        # search for the first safe direction
        for d in preferred_directions:
            next_pos = from_pos + d
            if next_pos in perception.visible_cells and perception.visible_cells[next_pos] not in (
            CellType.WALL, CellType.HAZARD):
                return d

        # No safe direction
        return None

    def _random_valid_move(self, visible_cells: Dict[Position, CellType],
                           current_pos: Position) -> Action:
        """ Select a random valid movement action"""
        valid_dirs = [
            d for d in Direction
            if (new_pos := current_pos + d) in visible_cells and
               visible_cells[new_pos] not in (CellType.WALL, CellType.HAZARD)
        ]
        if not valid_dirs:
            return Action.WAIT
        return self._direction_to_action(random.choice(valid_dirs))


class ModelBasedReflexAgent(Agent):
    """
    Model-Based Reflex Agent with communication capabilities.
    """

    def __init__(self, name: str):
        super().__init__(name)
        # Initialize internal world model data structures
        self.visited_positions: Set[Position] = set()
        self.known_resources: Set[Position] = set()
        self.known_goals: Set[Position] = set()
        self.known_hazards: Set[Position] = set()
        self.known_walls: Set[Position] = set()
        self.frontier: Set[Position] = set()

        self.rule_counts = defaultdict(int)
        self.last_perception: Optional[Perception] = None
        self.time_step = 0
        self.last_seen_agents: Set[int] = set()

    def _neighbors(self, pos: Position) -> List[Position]:
        return [pos + d for d in Direction]

    def _prepare_knowledge_message(self) -> str:
        """Serializes the agent's knowledge into a JSON string."""
        knowledge = {
            "walls": [asdict(p) for p in self.known_walls],
            "resources": [asdict(p) for p in self.known_resources],
            "goals": [asdict(p) for p in self.known_goals],
            "hazards": [asdict(p) for p in self.known_hazards],
            "visited": [asdict(p) for p in self.visited_positions]
        }
        return json.dumps(knowledge)

    def decide_action(self, perception: Perception) -> Tuple[Action, str]:
        """
        Implement model-based decision making with communication.
        """
        self.time_step += 1
        self.last_perception = perception
        self._update_world_model(perception)
        pos = perception.position

        # Staggered communication to avoid all agents broadcasting at once
        current_seen_ids = set(perception.visible_agents.keys())
        newly_seen = current_seen_ids - self.last_seen_agents
        if newly_seen:
            self.last_seen_agents = current_seen_ids
            self.rule_counts['communicate'] += 1
            return Action.COMMUNICATE, self._prepare_knowledge_message()

        if not self.known_resources and not self.frontier and not perception.has_resource:
            self.rule_counts['wait_idle'] += 1
            return Action.WAIT, "Exploration complete, no resources. Conserving energy."

        if self._is_in_immediate_danger(pos):
            self.rule_counts['emergency_avoidance'] += 1
            return self._emergency_avoidance(pos)

        if perception.has_resource:
            if pos in self.known_goals:
                self.rule_counts['drop_at_goal'] += 1
                return Action.DROP, f"Dropping resource at known goal {pos}"

        elif pos in self.known_resources:
            self.rule_counts['pickup'] += 1
            self.known_resources.discard(pos)
            return Action.PICKUP, f"Picking up resource at {pos}"

        if perception.has_resource:
            action, reason = self._move_toward_known_target(pos, self.known_goals, "goal")
            if action:
                self.rule_counts['to_goal'] += 1
                return action, reason

        if self.known_resources:
            action, reason = self._move_toward_known_target(pos, self.known_resources, "resource")
            if action:
                self.rule_counts['to_resource'] += 1
                return action, reason

        if self.frontier:
            action, reason = self._move_toward_known_target(pos, self.frontier, "frontier")
            if action:
                self.rule_counts['explore'] += 1
                return action, f"Exploring toward nearest frontier at {reason.split()[-1]}"

        return self._intelligent_exploration(pos)

    def _update_world_model(self, perception: Perception) -> None:
        """
        Update internal world model with new perceptual and communicated information.
        """
        current_pos = perception.position

        self.visited_positions.add(current_pos)

        self.frontier.discard(current_pos)
        for nbr in self._neighbors(current_pos):
            self.frontier.discard(nbr)

        for pos, cell_type in perception.visible_cells.items():
            self.visited_positions.add(pos)

            if cell_type == CellType.WALL:
                self.known_walls.add(pos)
                self.frontier.discard(pos)

            elif cell_type == CellType.HAZARD:
                self.known_hazards.add(pos)
                self.frontier.discard(pos)

            elif cell_type == CellType.RESOURCE:
                self.known_resources.add(pos)

            elif cell_type == CellType.GOAL:
                self.known_goals.add(pos)

            if cell_type not in (CellType.WALL, CellType.HAZARD):
                for neighbor in self._neighbors(pos):
                    if neighbor not in self.visited_positions and neighbor not in self.known_walls:
                        self.frontier.add(neighbor)

        for msg in perception.messages:
            if msg.sender_id != self.agent_id:
                try:
                    shared_knowledge = json.loads(msg.content)
                    self.known_walls.update(Position(**p) for p in shared_knowledge.get("walls", []))
                    self.known_resources.update(Position(**p) for p in shared_knowledge.get("resources", []))
                    self.known_goals.update(Position(**p) for p in shared_knowledge.get("goals", []))
                    self.known_hazards.update(Position(**p) for p in shared_knowledge.get("hazards", []))

                    # Add their visited nodes to our frontier if we haven't been there
                    their_visited = {Position(**p) for p in shared_knowledge.get("visited", [])}
                    for p in their_visited:
                        for nbr in self._neighbors(p):
                            if nbr not in self.visited_positions and nbr not in self.known_walls:
                                self.frontier.add(nbr)


                except (json.JSONDecodeError, TypeError):
                    pass

        visible_resource_positions = {p for p, t in perception.visible_cells.items() if t == CellType.RESOURCE}
        resources_in_view = self.known_resources.intersection(perception.visible_cells.keys())
        resources_to_remove = resources_in_view - visible_resource_positions
        self.known_resources -= resources_to_remove

    def _is_in_immediate_danger(self, current_pos: Position) -> bool:
        return current_pos in self.known_hazards

    def _get_safe_moves(self, current_pos: Position, occupied: Set[Position]) -> List[Direction]:
        safe_dirs = []
        for d in Direction:
            new_pos = current_pos + d
            if (new_pos not in self.known_walls
                    and new_pos not in self.known_hazards
                    and new_pos not in occupied):
                safe_dirs.append(d)
        return safe_dirs

    def _emergency_avoidance(self, current_pos: Position) -> Tuple[Action, str]:
        occupied = set(self.last_perception.visible_agents.values())
        safe_moves = self._get_safe_moves(current_pos, occupied)

        if safe_moves:
            direction = random.choice(safe_moves)
            return self._direction_to_action(direction), f"Avoiding hazard at {current_pos}"

        return Action.WAIT, f"No safe moves available from hazard at {current_pos}"

    def _move_toward_known_target(self, current_pos: Position, targets: Set[Position], target_type: str) -> Tuple[
        Optional[Action], str]:
        occupied = set(self.last_perception.visible_agents.values())

        if not targets:
            return None, f"No known {target_type}s"

        closest_target = min(targets, key=lambda pos: abs(pos.x - current_pos.x) + abs(pos.y - current_pos.y))
        direction = self._get_direction_toward(current_pos, closest_target)

        if direction:
            safe_moves = self._get_safe_moves(current_pos, occupied)
            if direction in safe_moves:
                return self._direction_to_action(direction), f"Moving toward known {target_type} at {closest_target}"

        safe_moves = self._get_safe_moves(current_pos, occupied)
        if safe_moves:
            return self._direction_to_action(
                random.choice(safe_moves)), f"Path to {closest_target} blocked, moving randomly"

        return None, f"No valid move toward {target_type} at {closest_target}"

    def _intelligent_exploration(self, current_pos: Position) -> Tuple[Action, str]:
        occupied = set(self.last_perception.visible_agents.values())
        safe_moves = self._get_safe_moves(current_pos, occupied)
        unvisited_dirs = [d for d in safe_moves if (current_pos + d) not in self.visited_positions]

        if unvisited_dirs:
            direction = random.choice(unvisited_dirs)
            return self._direction_to_action(direction), f"Exploring unvisited neighbor at {current_pos + direction}"
        if safe_moves:
            direction = random.choice(safe_moves)
            return self._direction_to_action(direction), f"Moving randomly to safe known cell {current_pos + direction}"

        return Action.WAIT, f"No valid moves for exploration at {current_pos}"

    def _direction_to_action(self, direction: Direction) -> Action:
        mapping = {Direction.NORTH: Action.MOVE_NORTH, Direction.SOUTH: Action.MOVE_SOUTH,
                   Direction.EAST: Action.MOVE_EAST, Direction.WEST: Action.MOVE_WEST}
        return mapping[direction]

    def _get_direction_toward(self, from_pos: Position, to_pos: Position) -> Optional[Direction]:
        dx = to_pos.x - from_pos.x
        dy = to_pos.y - from_pos.y
        if dx == 0 and dy == 0: return None
        if abs(dx) > abs(dy):
            return Direction.EAST if dx > 0 else Direction.WEST
        else:
            return Direction.SOUTH if dy > 0 else Direction.NORTH

    def reset(self):
        super().reset()
        self.visited_positions.clear()
        self.known_resources.clear()
        self.known_goals.clear()
        self.known_hazards.clear()
        self.known_walls.clear()
        self.rule_counts.clear()
        self.frontier.clear()
        self.last_perception = None
        self.time_step = 0


@dataclass
class PlanStep:
    """Represents a single step in an agent's plan"""
    action: Action
    target_position: Position
    purpose: str
    estimated_cost: float = 1.0


class GoalBasedAgent(Agent):
    """
    Goal-Based Agent with A* planning, communication, and deadlock avoidance.
    """


    def __init__(self, name: str):
        super().__init__(name)
        self.visited_positions: Set[Position] = set()
        self.known_resources: Set[Position] = set()
        self.known_goals: Set[Position] = set()
        self.known_hazards: Set[Position] = set()
        self.known_walls: Set[Position] = set()
        self.frontier: Set[Position] = set()

        self.current_plan: List[PlanStep] = []
        self.rule_counts = defaultdict(int)
        self.last_perception: Optional[Perception] = None
        self.current_goal: Optional[Tuple[str, Position]] = None
        self.time_step = 0
        self.last_seen_agents: Set[int] = set()

    def _prepare_knowledge_message(self) -> str:
        """Serializes the agent's knowledge into a JSON string."""
        knowledge = {
            "walls": [asdict(p) for p in self.known_walls],
            "resources": [asdict(p) for p in self.known_resources],
            "goals": [asdict(p) for p in self.known_goals],
            "hazards": [asdict(p) for p in self.known_hazards],
            "visited": [asdict(p) for p in self.visited_positions]
        }
        return json.dumps(knowledge)

    def decide_action(self, perception: Perception) -> Tuple[Action, str]:
        self.time_step += 1
        self.last_perception = perception
        self._update_world_model(perception)
        pos = perception.position

        # Communicate only once when a new agent is seen
        current_seen = set(perception.visible_agents.keys())
        new_agents = current_seen - self.last_seen_agents
        if new_agents:
            self.last_seen_agents = current_seen
            self.rule_counts['communicate'] += 1
            return Action.COMMUNICATE, self._prepare_knowledge_message()

        # Idle if nothing left
        if not perception.has_resource and not self.known_resources and not self.frontier:
            self.rule_counts['wait_idle'] += 1
            return Action.WAIT, "Mission complete. Conserving energy."

        # Avoid hazard
        if pos in self.known_hazards:
            self.current_plan.clear()
            self.current_goal = None
            self.rule_counts['avoid_hazard'] += 1
            return self._emergency_avoidance(pos, perception.visible_cells)

        # Pickup resource
        if perception.visible_cells.get(pos) == CellType.RESOURCE and not perception.has_resource:
            self.current_plan.clear()
            self.current_goal = None
            self.rule_counts['pickup'] += 1
            return Action.PICKUP, f"Picking up resource at {pos}"

        # Execute or replan
        if self.current_plan and self._is_plan_valid(perception):
            self.rule_counts['execute_plan'] += 1
            return self._execute_plan_step(pos)

        plan = self._create_new_plan(pos, perception)
        if plan:
            self.current_plan = plan
            self.rule_counts['create_plan'] += 1
            return self._execute_plan_step(pos)

        self.rule_counts['planning_failed_wait'] += 1
        return Action.WAIT, "Planning failed or no goals available. Reassessing."

    def _update_world_model(self, perception: Perception):
        current_pos = perception.position

        self.visited_positions.add(current_pos)
        self.frontier.discard(current_pos)

        for nbr in self._neighbors(current_pos):
            self.frontier.discard(nbr)

        for pos, cell_type in perception.visible_cells.items():
            self.visited_positions.add(pos)

            if cell_type == CellType.WALL:
                self.known_walls.add(pos)
                self.frontier.discard(pos)

            elif cell_type == CellType.HAZARD:
                self.known_hazards.add(pos)
                self.frontier.discard(pos)

            elif cell_type == CellType.RESOURCE:
                self.known_resources.add(pos)

            elif cell_type == CellType.GOAL:
                self.known_goals.add(pos)

            if cell_type not in (CellType.WALL, CellType.HAZARD):
                for nbr in self._neighbors(pos):
                    if nbr not in self.visited_positions and nbr not in self.known_walls:
                        self.frontier.add(nbr)

        # process others' messages and expand their visited neighbors
        for msg in perception.messages:
            if msg.sender_id == self.agent_id:
                continue
            try:
                shared = json.loads(msg.content)
                self.known_walls.update(Position(**p) for p in shared.get('walls', []))
                self.known_resources.update(Position(**p) for p in shared.get('resources', []))
                self.known_goals.update(Position(**p) for p in shared.get('goals', []))
                self.known_hazards.update(Position(**p) for p in shared.get('hazards', []))
                their_visited = {Position(**p) for p in shared.get('visited', [])}
                for p in their_visited:
                    for nbr in self._neighbors(p):
                        if nbr not in self.visited_positions and nbr not in self.known_walls:
                            self.frontier.add(nbr)
            except:
                pass

        visible_res = {p for p, t in perception.visible_cells.items() if t == CellType.RESOURCE}
        in_view = self.known_resources & perception.visible_cells.keys()
        self.known_resources -= (in_view - visible_res)

    def _is_plan_valid(self, perception: Perception) -> bool:
        """
        Validates the current plan against new information, including other agents.
        """
        if not self.current_plan or not self.current_goal:
            return False

        # Rule 1: Check if the next step is blocked by another agent
        next_step_pos = self.current_plan[0].target_position
        if next_step_pos in perception.visible_agents.values():
            self.rule_counts['replan_agent_blocking'] += 1
            return False  # Path is blocked, replan

        # Rule 2: Check if goal is still valid
        goal_type, target_pos = self.current_goal
        if goal_type == "collect" and target_pos not in self.known_resources:
            self.rule_counts['replan_goal_disappeared'] += 1
            return False
        if goal_type == "deliver" and not perception.has_resource:
            return False

        # Rule 3: Check for new hazards on path
        for step in self.current_plan:
            if step.target_position in self.known_hazards:
                self.rule_counts['replan_hazard_on_path'] += 1
                return False
        return True

    def _create_new_plan(self, current_pos: Position, perception: Perception) -> Optional[List[PlanStep]]:
        """
        Creates a new plan by selecting the best goal and finding a path.
        """
        all_possible_goals = self._get_all_sorted_goals(current_pos, perception)
        if not all_possible_goals:
            return None

        visible_agent_positions = set(perception.visible_agents.values())

        for goal_type, target_pos, utility in all_possible_goals:
            if target_pos in self.known_hazards:
                continue

            path = self._find_path(current_pos, target_pos, visible_agent_positions)

            if path:
                self.current_goal = (goal_type, target_pos)
                plan = []
                pos = current_pos
                for next_pos in path:
                    direction = self._get_direction_toward(pos, next_pos)
                    if direction:
                        plan.append(PlanStep(
                            action=self._direction_to_action(direction),
                            target_position=next_pos,
                            purpose=f"Move toward {goal_type} at {target_pos}"
                        ))
                    pos = next_pos

                if goal_type == "collect":
                    plan.append(PlanStep(action=Action.PICKUP, target_position=target_pos, purpose="Pick up resource"))
                elif goal_type == "deliver":
                    plan.append(
                        PlanStep(action=Action.DROP, target_position=target_pos, purpose="Drop resource at goal"))

                return plan
            else:
                if goal_type == "explore":
                    self.frontier.discard(target_pos)
                elif goal_type == "collect":
                    self.known_resources.discard(target_pos)

        return None

    def _find_path(self, start: Position, goal: Position, occupied_positions: Set[Position]) -> Optional[List[Position]]:
        """A* pathfinding that treats other agents as temporary walls."""

        frontier_pq = [(0, start)]
        came_from: Dict[Position, Optional[Position]] = {start: None}
        cost_so_far: Dict[Position, float] = {start: 0}
        closed: Set[Position] = set()

        def get_cost(pos: Position) -> float:
            if pos in self.known_walls or pos in occupied_positions:
                return float('inf')
            if pos in self.known_hazards:
                return 5.0
            return 1.0

        while frontier_pq:
            _, current = heapq.heappop(frontier_pq)

            if current in closed:
                continue
            closed.add(current)

            if current == goal:
                path = []
                while current != start:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path

            for neighbor in self._neighbors(current):
                if (neighbor not in self.visited_positions
                        and neighbor not in self.frontier
                        and neighbor not in self.known_resources
                        and neighbor not in self.known_goals):
                    continue

                move_cost = get_cost(neighbor)
                if move_cost == float('inf'):
                    continue

                new_cost = cost_so_far[current] + move_cost
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + abs(goal.x - neighbor.x) + abs(goal.y - neighbor.y)
                    heapq.heappush(frontier_pq, (priority, neighbor))
                    came_from[neighbor] = current

        return None  # No path found

    def _execute_plan_step(self, current_pos: Position) -> Tuple[Action, str]:
        if not self.current_plan:
            return Action.WAIT, "Attempted to execute empty plan."
        next_step = self.current_plan.pop(0)
        return next_step.action, f"Executing plan: {next_step.purpose}"

    def _emergency_avoidance(self, current_pos: Position, visible_cells: Dict[Position, CellType]) -> Tuple[
        Action, str]:
        occupied = set(self.last_perception.visible_agents.values())
        valid_dirs = [
            d for d in Direction
            if ((new_pos := current_pos + d) in visible_cells
                and visible_cells[new_pos] not in (CellType.HAZARD, CellType.WALL)
                and new_pos not in occupied)
        ]
        if valid_dirs:
            return self._direction_to_action(random.choice(valid_dirs)), f"Avoiding hazard at {current_pos}"
        return Action.WAIT, f"No safe moves available from hazard at {current_pos}"

    def _get_all_sorted_goals(self, current_pos: Position, perception: Perception) -> List[Tuple[str, Position, float]]:
        candidates = []
        if perception.has_resource:
            for goal_pos in self.known_goals:
                dist = abs(goal_pos.x - current_pos.x) + abs(goal_pos.y - current_pos.y)
                utility = 20.0 / (dist + 1)
                candidates.append(("deliver", goal_pos, utility))
        # Goal: Collect a resource
        else:
            for res_pos in self.known_resources:
                dist = abs(res_pos.x - current_pos.x) + abs(res_pos.y - current_pos.y)
                utility = 10.0 / (dist + 1)
                candidates.append(("collect", res_pos, utility))

        # Goal: Explore
        if self.frontier:
            sorted_frontier = sorted(list(self.frontier),
                                     key=lambda p: abs(p.x - current_pos.x) + abs(p.y - current_pos.y))
            for frontier_cell in sorted_frontier[:5]:
                dist = abs(frontier_cell.x - current_pos.x) + abs(frontier_cell.y - current_pos.y)
                utility = 1.0 / (dist + 1)
                candidates.append(("explore", frontier_cell, utility))

        return sorted(candidates, key=lambda x: x[2], reverse=True)

    def _neighbors(self, pos: Position) -> List[Position]:
        return [pos + d for d in Direction]

    def _direction_to_action(self, direction: Direction) -> Action:
        mapping = {Direction.NORTH: Action.MOVE_NORTH, Direction.SOUTH: Action.MOVE_SOUTH,
                   Direction.EAST: Action.MOVE_EAST, Direction.WEST: Action.MOVE_WEST}
        return mapping[direction]

    def _get_direction_toward(self, from_pos: Position, to_pos: Position) -> Optional[Direction]:
        dx = to_pos.x - from_pos.x
        dy = to_pos.y - from_pos.y
        if dx == 0 and dy == 0: return None
        if abs(dx) >= abs(dy):
            return Direction.EAST if dx > 0 else Direction.WEST
        return Direction.SOUTH if dy > 0 else Direction.NORTH

    def reset(self):
        super().reset()
        self.visited_positions.clear()
        self.known_resources.clear()
        self.known_goals.clear()
        self.known_hazards.clear()
        self.known_walls.clear()
        self.current_plan.clear()
        self.rule_counts.clear()
        self.last_perception = None
        self.current_goal = None
        self.frontier.clear()
        self.time_step = 0

# ============================================================================
# EXPERIMENTAL FRAMEWORK - PROVIDED FOR TESTING
# ============================================================================

def run_pygame_visualization(env, max_steps=200, cell_size=40):
    """Run a pygame visualization of the environment simulation."""
    pygame.init()

    width, height = env.width * cell_size, env.height * cell_size
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("GridWorld Visualization")

    GRID_COLORS = {
        CellType.EMPTY: (255, 255, 255),  # white
        CellType.WALL: (0, 0, 0),  # black
        CellType.GOAL: (255, 255, 0),  # yellow
        CellType.RESOURCE: (0, 255, 0),  # green
        CellType.HAZARD: (255, 0, 0),  # red
    }

    AGENT_COLORS = {
        "SimpleReflexAgent": (52, 152, 219),  # Blue
        "ModelBasedReflexAgent": (46, 204, 113),  # Green
        "GoalBasedAgent": (155, 89, 182),  # Purple
        "default": (127, 140, 141)  # Grey for fallback
    }

    font = pygame.font.SysFont(None, 20)

    running = True
    for step in range(max_steps):
        if not running:
            break

        # Handle quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Environment step
        env.step()

        # --- Drawing ---
        screen.fill(GRID_COLORS[CellType.EMPTY])

        # Draw grid
        for y in range(env.height):
            for x in range(env.width):
                cell_type = env.grid[y, x]
                if cell_type != CellType.EMPTY:
                    color = GRID_COLORS.get(cell_type, (200, 200, 200))
                    pygame.draw.rect(screen, color, pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size))
                pygame.draw.rect(screen, (220, 220, 220),
                                 pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size), 1)

        # Draw agents as triangles
        for agent_id, agent in env.agents.items():
            # if env.agent_energy[agent_id] <= 0:
            #     continue  # Don't draw agents with no energy

            # Get agent's position and type
            pos = env.agent_positions[agent_id]
            agent_type_name = agent.__class__.__name__

            # Select color based on agent type
            agent_color = AGENT_COLORS.get(agent_type_name, AGENT_COLORS["default"])

            # Calculate triangle points for a nice visual representation
            center_x = pos.x * cell_size + cell_size // 2
            center_y = pos.y * cell_size + cell_size // 2
            radius = cell_size // 2.5

            # Points for an upward-pointing equilateral triangle
            p1 = (center_x, center_y - radius * 0.8)
            p2 = (center_x - radius, center_y + radius * 0.7)
            p3 = (center_x + radius, center_y + radius * 0.7)

            pygame.draw.polygon(screen, agent_color, [p1, p2, p3])

            # Draw energy level on the agent
            agent_energy = int(env.agent_energy[agent_id])
            energy_text = font.render(str(agent_energy), True, (255, 255, 255))  # White text for good contrast
            text_rect = energy_text.get_rect(center=(center_x, center_y))
            screen.blit(energy_text, text_rect)

        # Display simulation info on screen
        metrics = env.get_performance_metrics()
        info_text = f"Step: {step}  Collected: {metrics['Total Resources Collected']}  AvgEnergy: {metrics['Average Energy Remaining']:.1f}"
        text_surface = font.render(info_text, True, (255, 255, 255))
        screen.blit(text_surface, (5, 5))

        pygame.display.flip()
        if step % 10 == 0:
            metrics = env.get_performance_metrics()
            print(
                f"Step {step}: Resources collected: {metrics['Total Resources Collected']}, Energy: {metrics['Average Energy Remaining']:.1f}")
        pygame.time.delay(150)
    pygame.quit()



@dataclass
class ExperimentConfig:
    """Configuration for experimental scenarios"""
    name: str
    description: str
    grid_size: Tuple[int, int]
    num_agents: int
    num_resources: int
    num_goals: int
    num_hazards: int
    max_steps: int
    num_trials: int


@dataclass
class ExperimentResult:
    """Results from a single experimental trial"""
    config_name: str
    trial_number: int
    agent_type: str
    total_steps: int
    tasks_completed: int
    total_resources_collected: int
    average_energy_remaining: float
    collision_count: int
    success_rate: float
    efficiency_score: float


class ProjectTester:
    """
    Testing framework for student implementations
    Use this to test and analyze your agent implementations
    """

    def __init__(self):
        self.experiment_configs = [
            ExperimentConfig(
                name="simple_collection",
                description="Basic resource collection in open environment",
                grid_size=(8, 8),
                num_agents=2,
                num_resources=4,
                num_goals=2,
                num_hazards=0,
                max_steps=100,
                num_trials=5
            ),
            ExperimentConfig(
                name="maze_navigation",
                description="Resource collection in maze environment",
                grid_size=(10, 10),
                num_agents=1,
                num_resources=4,
                num_goals=2,
                num_hazards=3,
                max_steps=150,
                num_trials=5
            ),
            ExperimentConfig(
                name="competitive_collection",
                description="Multiple agents competing for limited resources",
                grid_size=(12, 12),
                num_agents=3,
                num_resources=3,
                num_goals=2,
                num_hazards=2,
                max_steps=200,
                num_trials=5
            ),
            ExperimentConfig(
                name="dense_hazards",
                description="Small grid with many hazards and limited resources",
                grid_size=(8, 8),
                num_agents=2,
                num_resources=2,
                num_goals=2,
                num_hazards=6,
                max_steps=120,
                num_trials=5
            ),
            ExperimentConfig(
                name="sparse_exploration",
                description="Large open grid for long exploration and communication testing",
                grid_size=(15, 15),
                num_agents=3,
                num_resources=5,
                num_goals=3,
                num_hazards=1,
                max_steps=300,
                num_trials=5
            ),
            ExperimentConfig(
                name="hazard_blocking_goals",
                description="Goals blocked by hazards, testing emergency avoidance",
                grid_size=(10, 10),
                num_agents=2,
                num_resources=3,
                num_goals=2,
                num_hazards=5,
                max_steps=200,
                num_trials=5
            )
        ]

        self.agent_types = {
            "SimpleReflex": SimpleReflexAgent,
            "ModelBased": ModelBasedReflexAgent,
            "GoalBased": GoalBasedAgent
        }

    def test_single_agent(self, agent_class, config_name: str = "competitive_collection", visualize=True):
        """Test a single agent implementation with optional visualization and deterministic randomness."""
        config = next((c for c in self.experiment_configs if c.name == config_name), None)
        if not config:
            print(f"Error: Config '{config_name}' not found.")
            return

        # Deterministic randomization based on scenario name
        random.seed(hash(config.name) % 2 ** 32)

        env = GridWorld(config.grid_size[0], config.grid_size[1], config.num_agents)

        # Add wall borders
        walls = [Position(x, 0) for x in range(env.width)] + \
                [Position(x, env.height - 1) for x in range(env.width)] + \
                [Position(0, y) for y in range(1, env.height - 1)] + \
                [Position(env.width - 1, y) for y in range(1, env.height - 1)]
        env.add_walls(walls)

        # All valid positions (not walls)
        all_positions = [
            Position(x, y)
            for x in range(1, env.width - 1)
            for y in range(1, env.height - 1)
        ]
        random.shuffle(all_positions)

        def pick_unique(n):
            picked = all_positions[:n]
            del all_positions[:n]
            return picked

        env.add_resources(pick_unique(config.num_resources))
        env.add_goals(pick_unique(config.num_goals))
        env.add_hazards(pick_unique(config.num_hazards))

        for i in range(config.num_agents):
            agent = agent_class(f"TestAgent{i + 1}")
            env.add_agent(agent, pick_unique(1)[0])

        # Run
        if visualize:
            run_pygame_visualization(env, max_steps=config.max_steps, cell_size=40)
        else:
            for step in range(config.max_steps):
                results = env.step()
                if step % 10 == 0:
                    metrics = env.get_performance_metrics()
                    print(
                        f"Step: {step}  Collected: {metrics['Total Resources Collected']}  AvgEnergy: {metrics['Average Energy Remaining']:.1f}"
                    )

        final_metrics = env.get_performance_metrics()
        print(f"\n--- Final Results for {agent_class.__name__} on {config.name} ---")
        for key, value in final_metrics.items():
            print(f"{key}: {value:.2f}")

        # Save log
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        filename = log_dir / f"{agent_class.__name__}_{config.name}_log.json"
        with open(filename, "w") as f:
            log_data = {
                "agent_type": agent_class.__name__,
                "config": asdict(config),
                "metrics": final_metrics,
                "logs": env.logs
            }
            json.dump(log_data, f, indent=2, default=lambda o: o.__dict__)
        print(f"Log file saved to {filename}")
        return final_metrics

    def run_comparison(self):
        """Compare all implemented agent types across all scenarios"""
        all_results = []
        for config in self.experiment_configs:
            print(f"\n{'=' * 20} SCENARIO: {config.name.upper()} {'=' * 20}")
            for agent_name, agent_class in self.agent_types.items():
                print(f"\n--- Testing {agent_name} ---")
                try:
                    # For comparison, we run headless for speed
                    result = self.test_single_agent(agent_class, config.name, visualize=False)
                    result['agent_type'] = agent_name
                    result['scenario'] = config.name
                    all_results.append(result)
                    print(f"✓ {agent_name} completed successfully.")
                except Exception as e:
                    print(f"✗ {agent_name} failed: {e}")
                    import traceback
                    traceback.print_exc()

        # Create and display a summary Results
        if all_results:
            # Convert results to DataFrame
            df = pd.DataFrame(all_results)
            print("\nستون‌های دیتافریم:\n", df.columns.tolist())
            # Set plot style
            sns.set(style="whitegrid")

            # List of key metrics to plot
            metrics_to_plot = [
                "Success Rate (%)",
                "Overall Efficiency Score",
                "Average Energy Remaining",
                "Collision Count",
                "Average Task Completion Time",
                "Exploration Coverage (%)"
            ]

            # Plot each metric
            for metric in metrics_to_plot:
                plt.figure(figsize=(10, 6))
                ax = sns.barplot(
                    x="scenario", y=metric, hue="agent_type", data=df, errorbar="sd", palette="Set2"
                )
                ax.set_title(f"{metric} by Agent Type and Scenario", fontsize=14)
                ax.set_ylabel(metric, fontsize=12)
                ax.set_xlabel("Scenario", fontsize=12)
                plt.legend(title="Agent Type")
                plt.tight_layout()
                plt.show()

# ============================================================================
# MAIN TESTING FUNCTION
# ============================================================================

def main():
    """
    Main function for testing your implementations

     Uncomment sections as you complete each agent implementation
    """

    print("Multi-Agent Systems Project - Testing Framework")
    print("=" * 60)

    tester = ProjectTester()

    print("\n1. Testing SimpleReflexAgent...")
    tester.test_single_agent(SimpleReflexAgent)

    print("\n2. Testing ModelBasedReflexAgent...")
    tester.test_single_agent(ModelBasedReflexAgent)

    print("\n3. Testing GoalBasedAgent...")
    tester.test_single_agent(GoalBasedAgent)

    # print("\n4. Running full comparison (this may take a moment)...")
    # tester.run_comparison()

    print("\nImplementation complete! Proceed to experimental analysis.")

if __name__ == "__main__":
    main()