# Multi-Agent Intelligent Systems Project

A complete AI project implementing and analyzing **three intelligent agent architectures** based on Russell & Norvig's taxonomy, using a **multi-agent gridworld simulation environment**. Developed in Python, this project demonstrates agent behavior ranging from simple reflex to sophisticated goal-based planning with A\*.

---

## 📃 Overview

This project provides hands-on implementation of:

* Simple Reflex Agent
* Model-Based Reflex Agent
* Goal-Based Agent with A\* Planning

Each agent operates in a 2D grid environment and is evaluated on efficiency, adaptability, and communication under various scenarios.

---

## 🤖 Features

* Fully modular Python implementation
* GridWorld environment with obstacles, hazards, resources, and goals
* Partial observability and limited agent perception
* Inter-agent communication (broadcast-based)
* Pygame visualization of agent behavior
* Built-in experiment and benchmarking framework

---

## 📊 Agent Architectures

### 1. Simple Reflex Agent

* Reacts instantly to local stimuli using prioritized condition-action rules
* No memory or internal model
* Best for simple environments

### 2. Model-Based Reflex Agent

* Maintains internal memory of walls, hazards, goals, and resources
* Shares knowledge with nearby agents
* Performs intelligent exploration using a frontier-based approach

### 3. Goal-Based Agent

* Uses A\* search algorithm for planning
* Selects goals based on utility (hazard avoidance, collection, delivery, exploration)
* Continuously replans in dynamic and multi-agent settings

---

## 🌐 SEO-Optimized Keywords

> Artificial Intelligence Agents, Reflex vs Goal-Based Agents, Multi-Agent Systems, A\* Pathfinding in Python, AI Agent Communication, AI Agent Planning, GridWorld AI Simulation, Intelligent Agent Architectures

---

## 📆 Experimental Scenarios

Run experiments across diverse scenarios:

| Scenario               | Grid Size | Agents | Resources | Hazards | Max Steps |
| ---------------------- | --------- | ------ | --------- | ------- | --------- |
| Simple Collection      | 8x8       | 2      | 4         | 0       | 100       |
| Maze Navigation        | 10x10     | 1      | 4         | 3       | 150       |
| Competitive Collection | 12x12     | 3      | 3         | 2       | 200       |
| Dense Hazards          | 8x8       | 2      | 2         | 6       | 120       |
| Sparse Exploration     | 15x15     | 3      | 5         | 1       | 300       |
| Hazard-Blocked Goals   | 10x10     | 2      | 3         | 5       | 200       |

Performance metrics include:

* ✅ Success Rate
* ⚡️ Energy Efficiency
* ⏱ Task Completion Time
* ⚖ Efficiency Score
* ❌ Collision Count
* 🔍 Exploration Coverage

---

## 🔧 Installation

```bash
# Clone repository
$ git clone https://github.com/your-username/multi-agent-simulation
$ cd multi-agent-simulation

# Install requirements
$ pip install -r requirements.txt
```

---

## ▶️ Usage

### Test individual agent:

```bash
$ python main.py
```

You can toggle visualization or run comparisons inside `main()`.

### Run all agent comparisons:

Uncomment `tester.run_comparison()` in `main()` to test all agents across all scenarios.

---

## 🎓 Learning Outcomes

* Understand agent types and the PEAS framework
* Apply planning algorithms like A\*
* Analyze trade-offs between reactivity and deliberation
* Explore inter-agent communication and distributed intelligence

---

## 🔍 Citation

This project is based on:

> Russell, S. & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.)


---

## 🌍 License

MIT License - See `LICENSE` file for details.
