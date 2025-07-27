# Multi-Agent-Intelligent-Systems-Project
Multi-agent gridworld project implementing Simple Reflex, Model-Based Reflex, and Goal-Based Agents. Features A* pathfinding, communication, and performance analysis in a configurable environment with visualization. Built in Python for AI course.

Overview
This project implements a multi-agent gridworld environment and three distinct agent architectures—Simple Reflex, Model-Based Reflex, and Goal-Based Agents—as part of the Artificial Intelligence course assignment. The project demonstrates the application of Russell and Norvig's agent architecture framework in a controlled, partially observable gridworld environment. Agents navigate to collect resources, deliver them to goal zones, avoid hazards, and manage energy constraints while optionally communicating with other agents.
The implementation is written in Python and includes a testing framework for performance evaluation across various scenarios, with visualization capabilities using Pygame and analytical tools using Matplotlib, Seaborn, and Pandas. This README provides instructions for setting up, running, and analyzing the project, along with an overview of its structure and functionality.

Table of Contents

Features
Prerequisites
Installation
Usage
Running Individual Agent Tests
Running Full Comparison
Visualization


Project Structure
Agent Architectures
Simple Reflex Agent
Model-Based Reflex Agent
Goal-Based Agent


Environment Description
Performance Metrics
Experimental Scenarios
Dependencies
Contributing
License
Acknowledgements


Features

Three Agent Architectures: Implementation of Simple Reflex, Model-Based Reflex, and Goal-Based agents with distinct decision-making strategies.
GridWorld Environment: A configurable 2D grid with walls, resources, goals, hazards, and partial observability (5x5 perception range).
A Pathfinding: Goal-Based Agent uses A algorithm for optimal navigation, considering obstacles and hazards.
Communication System: Model-Based and Goal-Based agents support inter-agent communication to share world knowledge.
Visualization: Pygame-based visualization for real-time observation of agent behavior.
Performance Analysis: Comprehensive metrics (success rate, efficiency, energy usage, etc.) with statistical analysis and visualizations using Matplotlib and Seaborn.
Testing Framework: Automated testing across multiple scenarios with logging and result aggregation.
Modular Code: Well-structured, maintainable codebase with clear separation of concerns and robust error handling.


Prerequisites

Python: Version 3.8 or higher
Operating System: Windows, macOS, or Linux
Dependencies: See Dependencies for required Python packages


Installation

Clone the Repository:
git clone https://github.com/your-username/multi-agent-intelligent-systems.git
cd multi-agent-intelligent-systems


Create a Virtual Environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt


Verify Installation:Run the main script to ensure all dependencies are correctly installed:
python main.py




Usage
The project provides a flexible testing framework to evaluate agent performance in various scenarios. You can run individual agent tests with or without visualization or perform a full comparison across all agent types and scenarios.
Running Individual Agent Tests
To test a single agent type in a specific scenario with visualization:
python main.py

This runs the default test sequence, evaluating SimpleReflexAgent, ModelBasedReflexAgent, and GoalBasedAgent in the "competitive_collection" scenario with Pygame visualization.
To test a specific agent in a specific scenario without visualization:
from main import ProjectTester

tester = ProjectTester()
tester.test_single_agent(SimpleReflexAgent, config_name="simple_collection", visualize=False)

Available scenarios: simple_collection, maze_navigation, competitive_collection, dense_hazards, sparse_exploration, hazard_blocking_goals.
Running Full Comparison
To compare all agent types across all scenarios (headless mode for speed):
from main import ProjectTester

tester = ProjectTester()
tester.run_comparison()

This generates performance metrics and visualizations (bar plots) for each metric across scenarios and agent types.
Visualization
The project includes a Pygame-based visualization to observe agent behavior in real-time:

Grid: Displays cells (empty, walls, goals, resources, hazards) with distinct colors.
Agents: Represented as colored triangles with energy levels displayed.
Metrics: Real-time display of step count, resources collected, and average energy.

To enable visualization, ensure visualize=True when calling test_single_agent.

Project Structure
multi-agent-intelligent-systems/
├── logs/                    # Log files for each test run
├── main.py                  # Main project file with agent implementations and testing framework
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation


logs/: Stores JSON log files for each test run, including metrics and agent actions.
main.py: Contains the GridWorld environment, agent implementations (SimpleReflexAgent, ModelBasedReflexAgent, GoalBasedAgent), and the ProjectTester class for experimentation.
requirements.txt: Lists required Python packages.
README.md: This file, providing project documentation.


Agent Architectures
Simple Reflex Agent

Description: A reactive agent using condition-action rules with no internal state.
Behavior:
Avoid hazards (move to safe adjacent cell).
Pick up resources if present.
Move toward goals if carrying a resource.
Move toward visible resources if not carrying.
Explore randomly otherwise.


Strengths: Fast decision-making, low computational overhead.
Weaknesses: Limited to immediate perceptions, prone to local optima.

Model-Based Reflex Agent

Description: Maintains an internal world model (walls, resources, goals, hazards, visited positions) and supports inter-agent communication.
Behavior:
Avoid hazards using spatial memory.
Pick up resources opportunistically.
Navigate to known goals when carrying resources.
Move toward known resources.
Explore unvisited areas systematically.


Strengths: Improved navigation through memory, effective in partially observable environments.
Weaknesses: Higher computational overhead due to model maintenance.

Goal-Based Agent

Description: Uses A* pathfinding and utility-based goal selection for deliberative planning, with plan validation and adaptation.
Behavior:
Avoid hazards reactively.
Pick up resources opportunistically.
Execute or replan paths to goals/resources using A*.
Select goals based on utility (distance-adjusted).
Explore unvisited areas if no known targets.


Strengths: Optimal paths, adaptive to dynamic changes, effective in complex scenarios.
Weaknesses: Highest computational overhead due to planning.


Environment Description

Grid: Configurable 2D grid (e.g., 8x8, 10x10, 12x12).
Cell Types:
Empty: Navigable space.
Walls: Impassable obstacles.
Goals: Zones for resource delivery.
Resources: Collectible items.
Hazards: Energy-depleting areas.


Agent Properties:
Start with 100 energy points.
Consume energy per action (e.g., 1.0 for movement, 0.5 for pickup/drop, 4.0 extra on hazards).
Perceive a 5x5 neighborhood.
Can communicate knowledge (walls, resources, goals, hazards).


Objective: Collect resources and deliver them to goals while managing energy and avoiding hazards.


Performance Metrics
The testing framework collects the following metrics:

Total Resources Collected: Number of resources delivered to goals.
Success Rate (%): Percentage of initial resources collected.
Average Energy Remaining: Mean energy across agents at the end.
Total Steps: Number of simulation steps taken.
Collision Count: Number of attempted invalid moves.
Exploration Coverage (%): Percentage of navigable cells visited.
Average Task Completion Time: Mean steps to deliver a resource.
Energy Efficiency: Resources collected per unit of energy.
Overall Efficiency Score: Composite score balancing resources, time, energy, and collisions.


Experimental Scenarios
The project includes six predefined scenarios to test agent performance:

Simple Collection: 8x8 grid, 2 agents, 4 resources, 2 goals, no hazards, 100 steps.
Maze Navigation: 10x10 grid, 1 agent, 4 resources, 2 goals, 3 hazards, 150 steps.
Competitive Collection: 12x12 grid, 3 agents, 3 resources, 2 goals, 2 hazards, 200 steps.
Dense Hazards: 8x8 grid, 2 agents, 2 resources, 2 goals, 6 hazards, 120 steps.
Sparse Exploration: 15x15 grid, 3 agents, 5 resources, 3 goals, 1 hazard, 300 steps.
Hazard Blocking Goals: 10x10 grid, 2 agents, 3 resources, 2 goals, 5 hazards, 200 steps.

Each scenario is run for 5 trials to compute means and standard deviations.

Dependencies
The project requires the following Python packages, listed in requirements.txt:
numpy>=1.21.0
matplotlib>=3.4.0
pandas>=1.3.0
seaborn>=0.11.0
pygame>=2.0.0

Install them using:
pip install -r requirements.txt


Contributing
This project is an individual academic assignment, and direct contributions are not permitted as per the academic integrity policy. However, you may discuss general concepts and debugging strategies with peers, ensuring all code remains original. For suggestions or issues, please contact the project author or course instructor.

License
This project is for educational purposes only and is not distributed under an open-source license. All code is proprietary to the project author and may not be shared or reused without permission.

Acknowledgements

Instructor: Dr. Mahdi Eftekhari for guidance and project design.
References: Russell, S. J., & Norvig, P. (2020). Artificial Intelligence: A Modern Approach (4th ed.).
Tools: Python, NumPy, Matplotlib, Seaborn, Pygame for implementation and visualization.


Built with :heart: for the Artificial Intelligence course, 2025.
