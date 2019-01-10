# Team-based Multi-agent Reinforcement Learning
[Research Report](report/CSE293_Report_Team-based_MARL.pdf)


## Abstract

In multi-agent reinforcement learning (MARL), differentiating between
agent intelligence and organization intelligence may hold the key to major
breakthroughs.

This project separates the encoding of agent intelligence from organization
intelligence. Agents are programmed with a simple naive algorithm, but they are
organized under teams and provided with US versus THEM context. The
organization intelligence is separately encoded in the team’s culture, which
determines how team rewards are doled out to its agents on top of the
environmental reward they gather during training.

With the separation of agent and organization intelligences, the methodology
becomes mathematically and computationally simple. It can scale easily with the
number of agents and teams and it enables teams of agents to achieve a wide range
of desired results and behaviors with only slight changes to the team culture and
no change to the agents’ policy algorithm.

The new approach enables teams of agents to easily exceed the performance of
agents trained under “state-of-the art” MARL algorithms. In addition, the use of
team reward in culture can lead to agent specialization, which enables a team of
specialized agents to build a dominating strategy to a game which is previously
intransitive to multiple individual agents.

## Installation

`pip install -r requirements.txt`
