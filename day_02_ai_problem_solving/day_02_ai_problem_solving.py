"""
AI PROBLEM SOLVING
Intelligent Agents, Rational Agents, Environments, State, Actions, Goals, Utility

This program is a detailed, executable study guide for foundational AI
problem solving and intelligent-agent concepts.

The examples progress from basic concepts to more advanced models:

1. What is Artificial Intelligence?
2. Intelligent agents
3. Sensors, actuators, percepts and actions
4. Rational agents
5. Performance measures
6. Task environments
7. PEAS representation
8. Environment properties
9. States
10. State transitions
11. Actions
12. Goals
13. Goal-based agents
14. Utility
15. Utility-based agents
16. Search and problem formulation
17. State-space representation
18. Breadth-first search
19. Depth-first search
20. Uniform-cost search
21. Greedy best-first search
22. A* search
23. Heuristics
24. Deterministic and stochastic environments
25. Fully and partially observable environments
26. Static and dynamic environments
27. Episodic and sequential environments
28. Discrete and continuous environments
29. Known and unknown environments
30. Agent architectures
31. Reflex agents
32. Model-based agents
33. Goal-based agents
34. Utility-based agents
35. Learning agents
36. Rationality versus omniscience
37. Autonomy
38. Bounded rationality
39. Decision making under uncertainty
40. Expected utility
41. Trade-offs between goals
42. Multi-objective decision making
43. State representation
44. State abstraction
45. Search-space complexity
46. Agent problem-solving cycle
47. Integrated example

The program uses only Python's standard library.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional, Callable
from collections import deque
import heapq
import math
import random


# ============================================================
# SECTION 1
# ARTIFICIAL INTELLIGENCE AND PROBLEM SOLVING
# ============================================================

print("=" * 80)
print("AI PROBLEM SOLVING")
print("=" * 80)

print("""
Artificial Intelligence is concerned with building systems capable of
perceiving their surroundings, reasoning about available possibilities,
making decisions, taking actions and, in many cases, learning from experience.

An important way to study AI is through the concept of an AGENT.

An agent is an entity that:
    1. receives information from an environment,
    2. processes that information,
    3. chooses an action,
    4. performs that action,
    5. receives a new percept,
    6. continues the cycle.

The central abstraction is:

        ENVIRONMENT
             |
             v
          SENSORS
             |
             v
           AGENT
             |
             v
         ACTUATORS
             |
             v
        ENVIRONMENT
""")


# ============================================================
# SECTION 2
# AGENTS
# ============================================================

class Agent:
    """
    Basic conceptual model of an agent.

    A real intelligent agent can be much more complicated, but the
    fundamental abstraction is that an agent maps percept history to action.
    """

    def __init__(self, name: str):
        self.name = name
        self.percept_history = []

    def perceive(self, percept: Any):
        self.percept_history.append(percept)

    def choose_action(self, percept: Any) -> str:
        raise NotImplementedError


class SimpleReflexAgent(Agent):
    """
    A simple reflex agent chooses actions using the current percept only.

    It does not explicitly reason about long-term goals or utility.
    """

    def choose_action(self, percept: str) -> str:
        self.perceive(percept)

        if percept == "dirty":
            return "clean"

        if percept == "obstacle":
            return "turn"

        if percept == "danger":
            return "move_away"

        return "do_nothing"


reflex_agent = SimpleReflexAgent("Basic Reflex Agent")

print("\n--- Simple Agent Example ---")

for percept in ["clean", "dirty", "obstacle", "danger"]:
    action = reflex_agent.choose_action(percept)
    print(f"Percept: {percept:10} -> Action: {action}")


# ============================================================
# SECTION 3
# PERCEPTS, SENSORS, ACTUATORS AND ACTIONS
# ============================================================

print("\n" + "=" * 80)
print("PERCEPTS, SENSORS, ACTUATORS AND ACTIONS")
print("=" * 80)

print("""
A percept is the information an agent receives at a particular moment.

A percept sequence is the complete history of percepts received so far.

A sensor is the mechanism through which an agent obtains percepts.

An actuator is the mechanism through which an agent affects the environment.

Examples:

    Robot:
        Sensors  -> camera, lidar, microphone
        Actuators -> motors, robotic arms

    Self-driving car:
        Sensors  -> cameras, radar, lidar, GPS
        Actuators -> steering, brakes, accelerator

    Software agent:
        Sensors  -> APIs, files, user input
        Actuators -> API calls, database operations, messages
""")


@dataclass
class Percept:
    time: int
    observations: Dict[str, Any]


@dataclass
class Action:
    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)


percept = Percept(
    time=1,
    observations={
        "location": "intersection",
        "traffic_light": "red",
        "pedestrian": True,
        "speed": 20,
    }
)

action = Action(
    name="brake",
    parameters={"strength": "moderate"}
)

print("Percept:", percept)
print("Action:", action)


# ============================================================
# SECTION 4
# RATIONAL AGENTS
# ============================================================

print("\n" + "=" * 80)
print("RATIONAL AGENTS")
print("=" * 80)

print("""
A rational agent chooses the action expected to maximize its performance
measure, given:

    - the percept sequence,
    - its knowledge,
    - the actions available to it,
    - the environment,
    - the performance measure.

Rationality does NOT mean:

    "always choosing the action that eventually turns out to be best."

It means:

    "choosing the action that is expected to produce the best outcome
     based on the information available at the time."

This distinction is fundamental.

An agent can make a rational decision and still receive a bad outcome.

Example:

A navigation agent sees two roads.

Road A historically has a 90% probability of taking 20 minutes.
Road B historically has a 70% probability of taking 15 minutes.

The rational choice depends on the performance measure and the consequences
associated with each outcome.

Rationality therefore depends on expectations, not perfect knowledge.
""")


def rational_choice(options: Dict[str, float]) -> str:
    """
    Select the action with the highest expected performance score.
    """
    return max(options, key=options.get)


choices = {
    "take_highway": 82.0,
    "take_city_road": 71.0,
    "wait": 40.0,
}

print("Available actions and expected performance:")

for action_name, score in choices.items():
    print(f"{action_name:20} -> {score}")

print("Rational choice:", rational_choice(choices))


# ============================================================
# SECTION 5
# PERFORMANCE MEASURE
# ============================================================

print("\n" + "=" * 80)
print("PERFORMANCE MEASURES")
print("=" * 80)

print("""
A performance measure defines what counts as success for an agent.

Different performance measures can cause the same environment to produce
different rational actions.

For a delivery robot, possible performance criteria include:

    - delivery accuracy
    - delivery time
    - battery consumption
    - safety
    - cost
    - customer satisfaction

A performance measure should be distinguished from the agent's internal
reward or utility representation.

The external performance measure evaluates behavior.

The internal utility function can help the agent compare possible outcomes.
""")


@dataclass
class PerformanceMeasure:
    safety_weight: float
    speed_weight: float
    energy_weight: float

    def score(self, safety: float, speed: float, energy: float) -> float:
        return (
            self.safety_weight * safety
            + self.speed_weight * speed
            + self.energy_weight * energy
        )


measure = PerformanceMeasure(
    safety_weight=0.50,
    speed_weight=0.30,
    energy_weight=0.20
)

print(
    "Performance score:",
    measure.score(
        safety=95,
        speed=80,
        energy=70
    )
)


# ============================================================
# SECTION 6
# PEAS
# ============================================================

print("\n" + "=" * 80)
print("PEAS: TASK ENVIRONMENT SPECIFICATION")
print("=" * 80)

print("""
PEAS is a framework for specifying an agent's task environment.

P = Performance measure
E = Environment
A = Actuators
S = Sensors

Example: autonomous taxi

Performance:
    - safety
    - passenger comfort
    - travel time
    - fuel/energy efficiency
    - legality

Environment:
    - roads
    - traffic
    - pedestrians
    - weather
    - other vehicles

Actuators:
    - steering
    - brakes
    - accelerator
    - indicators

Sensors:
    - cameras
    - radar
    - lidar
    - GPS
    - speed sensors
""")


@dataclass
class PEAS:
    performance: List[str]
    environment: List[str]
    actuators: List[str]
    sensors: List[str]

    def display(self):
        print("\nPERFORMANCE:")
        for item in self.performance:
            print("  -", item)

        print("\nENVIRONMENT:")
        for item in self.environment:
            print("  -", item)

        print("\nACTUATORS:")
        for item in self.actuators:
            print("  -", item)

        print("\nSENSORS:")
        for item in self.sensors:
            print("  -", item)


taxi = PEAS(
    performance=[
        "Safety",
        "Low travel time",
        "Passenger comfort",
        "Energy efficiency",
        "Legal compliance",
    ],
    environment=[
        "Roads",
        "Vehicles",
        "Pedestrians",
        "Traffic signals",
        "Weather",
    ],
    actuators=[
        "Steering",
        "Brake",
        "Accelerator",
        "Indicators",
    ],
    sensors=[
        "Camera",
        "Radar",
        "Lidar",
        "GPS",
    ],
)

taxi.display()


# ============================================================
# SECTION 7
# ENVIRONMENT PROPERTIES
# ============================================================

print("\n" + "=" * 80)
print("TASK ENVIRONMENT PROPERTIES")
print("=" * 80)

print("""
AI environments are often classified using several dimensions.

1. Fully observable vs partially observable

Fully observable:
    The agent can obtain all relevant information needed to make decisions.

Partially observable:
    Important information is hidden, noisy or inaccessible.

2. Deterministic vs stochastic

Deterministic:
    The next state is completely determined by the current state and action.

Stochastic:
    The same action can produce different outcomes.

3. Episodic vs sequential

Episodic:
    Each decision is relatively independent.

Sequential:
    Current actions affect future situations.

4. Static vs dynamic

Static:
    The environment does not change while the agent is deciding.

Dynamic:
    The environment can change while the agent is reasoning.

5. Discrete vs continuous

Discrete:
    States, actions or time steps can be represented using distinct values.

Continuous:
    Variables can take values from continuous ranges.

6. Single-agent vs multi-agent

Single-agent:
    The agent is the primary decision maker.

Multi-agent:
    Other agents can affect the outcome.

7. Known vs unknown

Known:
    The agent knows the rules or transition model.

Unknown:
    The agent must learn or infer how the environment behaves.
""")


@dataclass
class EnvironmentProperties:
    observable: str
    deterministic: str
    episodic: str
    dynamic: str
    action_space: str
    agents: str
    knowledge: str

    def describe(self):
        properties = {
            "Observability": self.observable,
            "Determinism": self.deterministic,
            "Interaction": self.episodic,
            "Change": self.dynamic,
            "Action space": self.action_space,
            "Number of agents": self.agents,
            "Knowledge": self.knowledge,
        }

        for key, value in properties.items():
            print(f"{key:20}: {value}")


chess_environment = EnvironmentProperties(
    observable="Fully observable",
    deterministic="Deterministic",
    episodic="Sequential",
    dynamic="Static",
    action_space="Discrete",
    agents="Multi-agent",
    knowledge="Known",
)

print("\nChess environment:")
chess_environment.describe()


# ============================================================
# SECTION 8
# STATE
# ============================================================

print("\n" + "=" * 80)
print("STATE")
print("=" * 80)

print("""
A state is a representation of the relevant condition of an environment
at a particular point in time.

Consider a simple grid world.

        0   1   2
      +---+---+---+
    0 |   |   |   |
      +---+---+---+
    1 |   | A |   |
      +---+---+---+
    2 |   | G |   |
      +---+---+---+

The agent's state might be represented as:

    (row=1, column=1)

A richer state could include:

    position
    battery level
    carrying status
    obstacles
    time
    known information

The correct state representation depends on the problem.

A state representation should contain information relevant to future
decision making while avoiding unnecessary detail when possible.
""")


@dataclass(frozen=True)
class GridState:
    row: int
    col: int


initial_state = GridState(1, 1)
goal_state = GridState(2, 1)

print("Initial state:", initial_state)
print("Goal state:", goal_state)


# ============================================================
# SECTION 9
# ACTIONS
# ============================================================

print("\n" + "=" * 80)
print("ACTIONS")
print("=" * 80)

print("""
An action is something the agent can perform.

For a grid-world agent:

    UP
    DOWN
    LEFT
    RIGHT

An action does not necessarily mean that the agent will achieve its
desired result.

For example:

    State: (1, 1)
    Action: RIGHT

may result in:

    (1, 2)

if no obstacle exists.

Actions are usually defined by an action model or transition model.
""")


ACTIONS = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}


def apply_action(
    state: GridState,
    action_name: str,
    rows: int,
    cols: int
) -> Optional[GridState]:

    if action_name not in ACTIONS:
        return None

    dr, dc = ACTIONS[action_name]

    new_row = state.row + dr
    new_col = state.col + dc

    if not (0 <= new_row < rows):
        return None

    if not (0 <= new_col < cols):
        return None

    return GridState(new_row, new_col)


state = GridState(1, 1)

for action_name in ACTIONS:
    result = apply_action(state, action_name, 3, 3)
    print(f"{action_name:6} -> {result}")


# ============================================================
# SECTION 10
# TRANSITION MODEL
# ============================================================

print("\n" + "=" * 80)
print("TRANSITION MODEL")
print("=" * 80)

print("""
A transition model describes what happens after an action is performed.

In a deterministic environment:

    RESULT(state, action) = next_state

Example:

    RESULT((1,1), RIGHT) = (1,2)

In a stochastic environment, the result may be represented as a
probability distribution:

    P(next_state | current_state, action)

For example:

    P(A | state, action) = 0.8
    P(B | state, action) = 0.2

This distinction becomes extremely important when moving from classical
search into decision making under uncertainty.
""")


def deterministic_transition(
    state: GridState,
    action_name: str
) -> Optional[GridState]:

    return apply_action(state, action_name, 5, 5)


print(
    "Transition:",
    deterministic_transition(GridState(2, 2), "RIGHT")
)


# ============================================================
# SECTION 11
# GOALS
# ============================================================

print("\n" + "=" * 80)
print("GOALS")
print("=" * 80)

print("""
A goal describes a desired condition.

In problem solving, a goal is often represented as a set of goal states.

Example:

Initial state:
    (0, 0)

Goal:
    (4, 4)

The agent's objective is to find actions that transform the initial
state into a state satisfying the goal condition.

A goal is not the same thing as an action.

Goal:
    Reach the airport.

Action:
    Drive east.

Goal:
    Deliver the package.

Action:
    Move to the customer's address.

Goals specify desired outcomes.
Actions specify ways to change the state.
""")


def is_goal(state: GridState, target: GridState) -> bool:
    return state == target


print(
    "Is goal?",
    is_goal(GridState(2, 1), GridState(2, 1))
)


# ============================================================
# SECTION 12
# UTILITY
# ============================================================

print("\n" + "=" * 80)
print("UTILITY")
print("=" * 80)

print("""
A goal only distinguishes desired states from undesired states.

Utility provides a richer way to compare outcomes.

Suppose a delivery agent has two possible outcomes:

    Outcome A:
        delivery time = 20 minutes
        cost = low
        safety = high

    Outcome B:
        delivery time = 10 minutes
        cost = high
        safety = medium

A binary goal may treat both as successful.

A utility function can distinguish between them.

Utility is a numerical representation of how desirable an outcome is.

Higher utility generally means a more preferred outcome.

A utility function can combine multiple criteria.
""")


@dataclass
class Outcome:
    name: str
    time_minutes: float
    cost: float
    safety: float
    comfort: float


def utility(
    outcome: Outcome,
    time_weight: float,
    cost_weight: float,
    safety_weight: float,
    comfort_weight: float
) -> float:

    # Lower time and cost are desirable.
    # Higher safety and comfort are desirable.

    return (
        -time_weight * outcome.time_minutes
        -cost_weight * outcome.cost
        +safety_weight * outcome.safety
        +comfort_weight * outcome.comfort
    )


outcome_a = Outcome(
    name="Fast Route",
    time_minutes=15,
    cost=20,
    safety=70,
    comfort=65
)

outcome_b = Outcome(
    name="Safe Route",
    time_minutes=25,
    cost=15,
    safety=95,
    comfort=90
)

for outcome in [outcome_a, outcome_b]:
    score = utility(
        outcome,
        time_weight=1.0,
        cost_weight=0.5,
        safety_weight=1.5,
        comfort_weight=0.5
    )

    print(f"{outcome.name:15} -> Utility = {score:.2f}")


# ============================================================
# SECTION 13
# GOAL VS UTILITY
# ============================================================

print("\n" + "=" * 80)
print("GOAL-BASED VS UTILITY-BASED DECISION MAKING")
print("=" * 80)

print("""
Goal-based reasoning asks:

    "Does this state satisfy the goal?"

Utility-based reasoning asks:

    "How desirable is this state compared with other possible states?"

Suppose the goal is:

    Reach the destination.

Three routes may all satisfy the goal:

    Route A -> 20 minutes
    Route B -> 35 minutes
    Route C -> 50 minutes

Goal-based reasoning may regard all three as successful.

Utility-based reasoning can rank them:

    A > B > C

Utility becomes particularly valuable when:

    - multiple goals exist,
    - goals conflict,
    - outcomes have different quality,
    - uncertainty exists,
    - trade-offs must be made.
""")


def goal_satisfied(destination_reached: bool) -> bool:
    return destination_reached


def route_utility(time: float, safety: float, cost: float) -> float:
    return (
        -2.0 * time
        + 3.0 * safety
        -1.0 * cost
    )


routes = {
    "A": {"time": 20, "safety": 90, "cost": 10},
    "B": {"time": 30, "safety": 98, "cost": 8},
    "C": {"time": 45, "safety": 100, "cost": 5},
}

for name, values in routes.items():
    score = route_utility(**values)
    print(f"Route {name}: utility = {score}")


# ============================================================
# SECTION 14
# PROBLEM FORMULATION
# ============================================================

print("\n" + "=" * 80)
print("PROBLEM FORMULATION")
print("=" * 80)

print("""
A classical AI search problem can be formulated using:

    1. Initial state
    2. Actions
    3. Transition model
    4. Goal test
    5. Path cost

This creates a state-space search problem.

For example, route planning:

    Initial state:
        Lucknow

    Actions:
        travel to connected city

    Transition:
        move to another city

    Goal:
        Delhi

    Path cost:
        distance, time, fuel or another measure

The distinction between state, action, goal and cost is essential.

STATE:
    Where or in what condition the agent currently is.

ACTION:
    What the agent can do.

GOAL:
    What condition the agent wants to achieve.

PATH COST:
    What it costs to execute a sequence of actions.
""")


@dataclass
class SearchProblem:
    initial_state: Any
    goal_test: Callable[[Any], bool]
    actions: Callable[[Any], List[Any]]
    result: Callable[[Any, Any], Any]
    step_cost: Callable[[Any, Any, Any], float]


# ============================================================
# SECTION 15
# STATE-SPACE GRAPH
# ============================================================

print("\n" + "=" * 80)
print("STATE-SPACE REPRESENTATION")
print("=" * 80)

print("""
A state-space graph represents:

    Nodes  -> states
    Edges  -> actions/transitions

Example:

        A
       / \
      B   C
      |   |
      D   E
       \ /
        G

The agent searches through this space to find a path from the initial
state to a goal state.

Important distinction:

A STATE is not necessarily the same as a NODE in an implementation.

A node in a search tree can contain:

    state
    parent
    action
    path cost
    depth

Two different search-tree nodes may represent the same underlying state
if they were reached through different paths.
""")


@dataclass
class Node:
    state: Any
    parent: Optional["Node"] = None
    action: Any = None
    path_cost: float = 0.0
    depth: int = 0

    def path(self) -> List[Any]:
        node = self
        result = []

        while node is not None:
            result.append(node.state)
            node = node.parent

        result.reverse()
        return result


graph = {
    "A": ["B", "C"],
    "B": ["D"],
    "C": ["E"],
    "D": ["G"],
    "E": ["G"],
    "G": [],
}

print("State-space graph:")
for state, successors in graph.items():
    print(f"{state} -> {successors}")


# ============================================================
# SECTION 16
# BREADTH-FIRST SEARCH
# ============================================================

print("\n" + "=" * 80)
print("BREADTH-FIRST SEARCH")
print("=" * 80)

print("""
Breadth-first search explores the shallowest nodes first.

It uses a FIFO queue.

Conceptually:

    Queue
    -----
    first in -> first out

BFS is complete under standard assumptions.

When every action has the same cost, BFS also finds a shallowest solution,
which means an optimal solution with respect to number of actions.

Time complexity is commonly described as:

    O(b^d)

Space complexity:

    O(b^d)

where:

    b = branching factor
    d = depth of the shallowest goal
""")


def breadth_first_search(
    graph: Dict[Any, List[Any]],
    start: Any,
    goal: Any
) -> Optional[List[Any]]:

    frontier = deque([
        Node(state=start)
    ])

    explored = set()

    while frontier:
        node = frontier.popleft()

        if node.state == goal:
            return node.path()

        if node.state in explored:
            continue

        explored.add(node.state)

        for child_state in graph.get(node.state, []):
            if child_state not in explored:
                child = Node(
                    state=child_state,
                    parent=node,
                    action=None,
                    path_cost=node.path_cost + 1,
                    depth=node.depth + 1
                )

                frontier.append(child)

    return None


bfs_result = breadth_first_search(graph, "A", "G")

print("BFS solution:", bfs_result)


# ============================================================
# SECTION 17
# DEPTH-FIRST SEARCH
# ============================================================

print("\n" + "=" * 80)
print("DEPTH-FIRST SEARCH")
print("=" * 80)

print("""
Depth-first search explores one branch as deeply as possible before
backtracking.

It is naturally implemented using a stack.

DFS can use significantly less memory than BFS in some search spaces.

Typical complexity:

    Time:
        O(b^m)

    Space:
        O(bm)

where m is the maximum depth.

DFS is not generally optimal.

In spaces with infinite depth or cycles, naive DFS can fail to find a
solution even when one exists.
""")


def depth_first_search(
    graph: Dict[Any, List[Any]],
    start: Any,
    goal: Any
) -> Optional[List[Any]]:

    frontier = [
        Node(state=start)
    ]

    explored = set()

    while frontier:
        node = frontier.pop()

        if node.state == goal:
            return node.path()

        if node.state in explored:
            continue

        explored.add(node.state)

        for child_state in reversed(graph.get(node.state, [])):
            if child_state not in explored:
                frontier.append(
                    Node(
                        state=child_state,
                        parent=node,
                        path_cost=node.path_cost + 1,
                        depth=node.depth + 1
                    )
                )

    return None


dfs_result = depth_first_search(graph, "A", "G")

print("DFS solution:", dfs_result)


# ============================================================
# SECTION 18
# UNIFORM-COST SEARCH
# ============================================================

print("\n" + "=" * 80)
print("UNIFORM-COST SEARCH")
print("=" * 80)

print("""
Uniform-cost search expands the node with the lowest path cost.

It is useful when actions have different costs.

Suppose:

    A -> B costs 2
    A -> C costs 5
    B -> G costs 10
    C -> G costs 2

The cheapest complete route is:

    A -> C -> G

with cost:

    5 + 2 = 7

while:

    A -> B -> G

costs:

    2 + 10 = 12

UCS therefore reasons about accumulated cost rather than depth.
""")


weighted_graph = {
    "A": [("B", 2), ("C", 5)],
    "B": [("G", 10)],
    "C": [("G", 2)],
    "G": [],
}


def uniform_cost_search(
    graph: Dict[Any, List[Tuple[Any, float]]],
    start: Any,
    goal: Any
) -> Optional[Tuple[List[Any], float]]:

    counter = 0

    frontier = [
        (0, counter, Node(state=start))
    ]

    best_cost = {start: 0}

    while frontier:
        cost, _, node = heapq.heappop(frontier)

        if node.state == goal:
            return node.path(), cost

        if cost > best_cost.get(node.state, math.inf):
            continue

        for child_state, step_cost in graph.get(node.state, []):
            new_cost = cost + step_cost

            if new_cost < best_cost.get(child_state, math.inf):
                best_cost[child_state] = new_cost
                counter += 1

                child = Node(
                    state=child_state,
                    parent=node,
                    path_cost=new_cost,
                    depth=node.depth + 1
                )

                heapq.heappush(
                    frontier,
                    (new_cost, counter, child)
                )

    return None


ucs_result = uniform_cost_search(weighted_graph, "A", "G")

print("UCS solution:", ucs_result)


# ============================================================
# SECTION 19
# HEURISTIC FUNCTIONS
# ============================================================

print("\n" + "=" * 80)
print("HEURISTICS")
print("=" * 80)

print("""
A heuristic function h(n) estimates the cost of reaching a goal from node n.

It represents problem-specific knowledge.

For example, in route planning:

    h(n) = straight-line distance from n to destination

A heuristic does not necessarily give the exact remaining cost.

If:

    h(n) = actual cheapest remaining cost

then the estimate is exact.

If:

    h(n) <= actual cheapest remaining cost

then the heuristic is admissible.

An admissible heuristic never overestimates the true cost to the goal.

Consistency is a stronger useful property:

    h(n) <= c(n,a,n') + h(n')

for every applicable action.

This is also called the triangle inequality property.
""")


def manhattan_distance(
    state: GridState,
    goal: GridState
) -> int:

    return abs(state.row - goal.row) + abs(state.col - goal.col)


print(
    "Manhattan heuristic:",
    manhattan_distance(
        GridState(1, 1),
        GridState(4, 5)
    )
)


# ============================================================
# SECTION 20
# GREEDY BEST-FIRST SEARCH
# ============================================================

print("\n" + "=" * 80)
print("GREEDY BEST-FIRST SEARCH")
print("=" * 80)

print("""
Greedy best-first search selects a node according to:

    f(n) = h(n)

It focuses on estimated distance to the goal.

It can be fast when the heuristic is informative.

It is not generally optimal because it ignores the path cost already
incurred.

A node that appears close to the goal may have required an extremely
expensive path to reach.
""")


# ============================================================
# SECTION 21
# A* SEARCH
# ============================================================

print("\n" + "=" * 80)
print("A* SEARCH")
print("=" * 80)

print("""
A* combines path cost and heuristic estimate.

The evaluation function is:

    f(n) = g(n) + h(n)

where:

    g(n) = cost from the initial state to n
    h(n) = estimated cost from n to a goal

Therefore:

    f(n) = estimated total solution cost through n

A* balances:

    "How much have I already spent?"

with:

    "How much do I expect to spend?"

Under standard conditions, A* is optimal when the heuristic is admissible
and the relevant assumptions about the search formulation hold.
""")


def a_star_grid(
    start: GridState,
    goal: GridState,
    rows: int,
    cols: int,
    obstacles: Optional[set] = None
) -> Optional[List[GridState]]:

    if obstacles is None:
        obstacles = set()

    counter = 0

    start_node = Node(state=start)

    frontier = [
        (
            manhattan_distance(start, goal),
            counter,
            start_node
        )
    ]

    best_g = {start: 0}

    while frontier:
        _, _, node = heapq.heappop(frontier)

        current = node.state

        if current == goal:
            return node.path()

        current_g = best_g[current]

        for action_name in ACTIONS:
            next_state = apply_action(
                current,
                action_name,
                rows,
                cols
            )

            if next_state is None:
                continue

            if next_state in obstacles:
                continue

            new_g = current_g + 1

            if new_g < best_g.get(next_state, math.inf):
                best_g[next_state] = new_g

                h = manhattan_distance(
                    next_state,
                    goal
                )

                f = new_g + h

                counter += 1

                child = Node(
                    state=next_state,
                    parent=node,
                    action=action_name,
                    path_cost=new_g,
                    depth=node.depth + 1
                )

                heapq.heappush(
                    frontier,
                    (f, counter, child)
                )

    return None


start = GridState(0, 0)
goal = GridState(4, 4)

obstacles = {
    GridState(1, 1),
    GridState(1, 2),
    GridState(2, 2),
    GridState(3, 2),
}

astar_path = a_star_grid(
    start,
    goal,
    rows=5,
    cols=5,
    obstacles=obstacles
)

print("A* path:")
print(astar_path)


# ============================================================
# SECTION 22
# AGENT TYPES
# ============================================================

print("\n" + "=" * 80)
print("TYPES OF AGENTS")
print("=" * 80)

print("""
Several standard agent architectures are useful for understanding
intelligent behavior.

1. Simple reflex agent

    Action depends on current percept.

2. Model-based reflex agent

    Maintains an internal state describing aspects of the world.

3. Goal-based agent

    Considers goals when selecting actions.

4. Utility-based agent

    Compares possible outcomes using utility.

5. Learning agent

    Improves behavior using experience.

These categories can be viewed as increasing levels of internal
representation and decision sophistication.
""")


class GoalBasedAgent(Agent):

    def __init__(self, name: str, goal: Any):
        super().__init__(name)
        self.goal = goal

    def choose_action(
        self,
        current_state: GridState
    ) -> str:

        if current_state.row < self.goal.row:
            return "DOWN"

        if current_state.row > self.goal.row:
            return "UP"

        if current_state.col < self.goal.col:
            return "RIGHT"

        if current_state.col > self.goal.col:
            return "LEFT"

        return "STOP"


goal_agent = GoalBasedAgent(
    "Grid Goal Agent",
    GridState(3, 3)
)

print(
    "Goal-based action:",
    goal_agent.choose_action(GridState(1, 1))
)


# ============================================================
# SECTION 23
# MODEL-BASED AGENT
# ============================================================

print("\n" + "=" * 80)
print("MODEL-BASED AGENT")
print("=" * 80)

print("""
A model-based agent maintains an internal representation of aspects of
the environment.

This is useful when the environment is partially observable.

Suppose a robot cannot see behind a wall.

A purely reflex-based system cannot directly observe what is behind the wall.

A model-based system can maintain an internal state such as:

    wall_position
    previously observed obstacle
    estimated robot position
    previous action

The internal model helps the agent reason about hidden aspects of the world.
""")


class ModelBasedAgent(Agent):

    def __init__(self, name: str):
        super().__init__(name)
        self.internal_state = {}

    def update_model(self, percept: Dict[str, Any]):
        self.internal_state.update(percept)

    def choose_action(self, percept: Dict[str, Any]) -> str:
        self.perceive(percept)
        self.update_model(percept)

        if percept.get("obstacle") is True:
            return "turn"

        if self.internal_state.get("battery", 100) < 20:
            return "recharge"

        return "move"


model_agent = ModelBasedAgent("Model Agent")

print(
    model_agent.choose_action(
        {
            "location": "room",
            "obstacle": False,
            "battery": 75
        }
    )
)

print("Internal state:", model_agent.internal_state)


# ============================================================
# SECTION 24
# UTILITY-BASED AGENT
# ============================================================

print("\n" + "=" * 80)
print("UTILITY-BASED AGENT")
print("=" * 80)

print("""
A utility-based agent evaluates possible outcomes and chooses an action
that leads to the highest expected utility.

This is useful when:

    - several outcomes are possible,
    - multiple goals compete,
    - there is uncertainty,
    - no single goal test adequately describes preference.
""")


class UtilityBasedAgent:

    def __init__(self, name: str):
        self.name = name

    def choose_action(
        self,
        expected_utilities: Dict[str, float]
    ) -> str:

        return max(
            expected_utilities,
            key=expected_utilities.get
        )


utility_agent = UtilityBasedAgent("Utility Agent")

possible_actions = {
    "route_A": 73.5,
    "route_B": 91.0,
    "route_C": 84.0,
}

print(
    "Chosen action:",
    utility_agent.choose_action(possible_actions)
)


# ============================================================
# SECTION 25
# RATIONALITY IS NOT OMNISCIENCE
# ============================================================

print("\n" + "=" * 80)
print("RATIONALITY VS OMNISCIENCE")
print("=" * 80)

print("""
Omniscience means knowing the actual outcome of every possible action.

A real agent is generally not omniscient.

Consider a weather-sensitive delivery decision.

At 9:00 AM:

    Action A appears to have expected utility = 80
    Action B appears to have expected utility = 70

The agent chooses A.

At 11:00 AM, unexpected weather makes A perform poorly.

The fact that A produced a poor result does not necessarily mean that
the original decision was irrational.

Rationality is evaluated relative to:

    - available information,
    - available actions,
    - knowledge,
    - computational limitations,
    - expected outcomes.

This is one of the most important distinctions in intelligent-agent theory.
""")


# ============================================================
# SECTION 26
# AUTONOMY
# ============================================================

print("\n" + "=" * 80)
print("AUTONOMY")
print("=" * 80)

print("""
An agent is autonomous to the extent that its behavior is determined by
its own experience rather than being completely controlled by its designer.

A completely fixed rule system has little autonomy.

A learning agent can adapt its behavior based on experience.

Autonomy does not mean independence from all external information.

An autonomous system can still use:

    - sensors,
    - instructions,
    - models,
    - policies,
    - external knowledge.

The important issue is the degree to which behavior is generated from
the agent's own learned or accumulated experience.
""")


# ============================================================
# SECTION 27
# BOUNDED RATIONALITY
# ============================================================

print("\n" + "=" * 80)
print("BOUNDED RATIONALITY")
print("=" * 80)

print("""
Perfect rational decision making can be computationally impossible.

An agent may have:

    - limited time,
    - limited memory,
    - limited computational power,
    - incomplete information.

This leads to bounded rationality.

A practically intelligent agent may choose the best action it can compute
within its available resources.

For example:

A chess engine cannot necessarily enumerate every possible continuation
of the game.

Instead, it uses:

    - search,
    - evaluation functions,
    - pruning,
    - heuristics,
    - time limits,
    - learned knowledge.

The resulting decision can be highly rational despite being computationally
limited.
""")


# ============================================================
# SECTION 28
# DETERMINISTIC VS STOCHASTIC ENVIRONMENT
# ============================================================

print("\n" + "=" * 80)
print("DETERMINISTIC VS STOCHASTIC")
print("=" * 80)

print("""
Deterministic:

    P(next_state | state, action) = 1

There is only one possible result.

Stochastic:

    multiple outcomes can have non-zero probability.

Example:

A robot attempts to move forward.

Possible outcomes:

    move successfully -> 0.90
    slip              -> 0.07
    obstacle detected -> 0.03

The agent must reason using probabilities.
""")


def stochastic_move(
    success_probability: float = 0.9
) -> str:

    r = random.random()

    if r < success_probability:
        return "success"

    return "failure"


random.seed(42)

for _ in range(5):
    print(stochastic_move())


# ============================================================
# SECTION 29
# EXPECTED UTILITY
# ============================================================

print("\n" + "=" * 80)
print("EXPECTED UTILITY")
print("=" * 80)

print("""
When an action can produce several outcomes, expected utility can be used.

The expected utility of an action is:

        EU(action)
        =
        sum over outcomes:
        P(outcome | action) * U(outcome)

Example:

Action A:

    0.8 probability -> utility 100
    0.2 probability -> utility 20

Expected utility:

    0.8(100) + 0.2(20)
    = 80 + 4
    = 84

Action B:

    0.5 probability -> utility 150
    0.5 probability -> utility 30

Expected utility:

    0.5(150) + 0.5(30)
    = 75 + 15
    = 90

A utility-maximizing agent would prefer B under this model.
""")


def expected_utility(
    outcomes: List[Tuple[float, float]]
) -> float:

    return sum(
        probability * utility_value
        for probability, utility_value in outcomes
    )


action_a = [
    (0.8, 100),
    (0.2, 20),
]

action_b = [
    (0.5, 150),
    (0.5, 30),
]

print("Expected utility A:", expected_utility(action_a))
print("Expected utility B:", expected_utility(action_b))


# ============================================================
# SECTION 30
# MULTI-OBJECTIVE UTILITY
# ============================================================

print("\n" + "=" * 80)
print("MULTI-OBJECTIVE DECISION MAKING")
print("=" * 80)

print("""
Real-world decisions rarely involve one objective.

An autonomous vehicle may care about:

    safety
    travel time
    energy
    comfort
    legality

These objectives can conflict.

A weighted utility model can combine them:

    U =
        w1 * safety
        + w2 * speed
        + w3 * comfort
        - w4 * cost

The weights express the relative importance assigned to each criterion.

Changing the weights can change the rational action.
""")


@dataclass
class DecisionOption:
    name: str
    safety: float
    speed: float
    comfort: float
    cost: float


def multi_objective_utility(
    option: DecisionOption,
    weights: Dict[str, float]
) -> float:

    return (
        weights["safety"] * option.safety
        + weights["speed"] * option.speed
        + weights["comfort"] * option.comfort
        - weights["cost"] * option.cost
    )


options = [
    DecisionOption(
        "Option A",
        safety=95,
        speed=70,
        comfort=80,
        cost=30
    ),
    DecisionOption(
        "Option B",
        safety=85,
        speed=95,
        comfort=70,
        cost=40
    ),
    DecisionOption(
        "Option C",
        safety=99,
        speed=60,
        comfort=95,
        cost=20
    ),
]

weights = {
    "safety": 0.5,
    "speed": 0.2,
    "comfort": 0.2,
    "cost": 0.1,
}

for option in options:
    score = multi_objective_utility(
        option,
        weights
    )

    print(
        f"{option.name}: {score:.2f}"
    )


# ============================================================
# SECTION 31
# PARTIALLY OBSERVABLE ENVIRONMENTS
# ============================================================

print("\n" + "=" * 80)
print("PARTIALLY OBSERVABLE ENVIRONMENTS")
print("=" * 80)

print("""
An environment is partially observable when the agent cannot directly
observe all relevant aspects of the true state.

Examples:

    - medical diagnosis
    - driving in fog
    - poker
    - robot navigation behind obstacles
    - financial decision making

The agent may maintain a belief state.

A belief state represents the agent's current information about possible
world states.

For example:

    State A -> probability 0.60
    State B -> probability 0.30
    State C -> probability 0.10

The agent does not know exactly which state is true.

It reasons over a distribution of possibilities.
""")


belief_state = {
    "state_A": 0.60,
    "state_B": 0.30,
    "state_C": 0.10,
}

print("Belief state:")

for state_name, probability in belief_state.items():
    print(
        f"{state_name}: {probability:.2f}"
    )

print(
    "Probability total:",
    sum(belief_state.values())
)


# ============================================================
# SECTION 32
# STATE ABSTRACTION
# ============================================================

print("\n" + "=" * 80)
print("STATE ABSTRACTION")
print("=" * 80)

print("""
A state representation can be extremely detailed.

Suppose a navigation problem records:

    exact GPS position
    exact speed
    exact acceleration
    tire pressure
    engine temperature
    weather
    traffic
    road condition
    time
    fuel

For a simple route-planning problem, much of this information may be
irrelevant.

An abstract state might simply be:

    current_city

Abstraction reduces the size of the search space.

But excessive abstraction can remove information necessary for correct
decision making.

Therefore state representation is a modeling decision.

A good representation preserves distinctions that matter to future actions
and outcomes.
""")


@dataclass(frozen=True)
class DetailedVehicleState:
    location: str
    speed: float
    fuel: float
    engine_temperature: float
    weather: str
    traffic: str


@dataclass(frozen=True)
class AbstractVehicleState:
    location: str


detailed = DetailedVehicleState(
    location="A",
    speed=60,
    fuel=70,
    engine_temperature=85,
    weather="clear",
    traffic="moderate"
)

abstract = AbstractVehicleState(
    location=detailed.location
)

print("Detailed state:", detailed)
print("Abstract state:", abstract)


# ============================================================
# SECTION 33
# SEARCH TREE VS STATE-SPACE GRAPH
# ============================================================

print("\n" + "=" * 80)
print("SEARCH TREE VS STATE-SPACE GRAPH")
print("=" * 80)

print("""
A search tree records different paths separately.

Suppose:

        A
       / \
      B   C
       \ /
        D

D can be reached through:

    A -> B -> D

or:

    A -> C -> D

A search tree can contain two separate nodes representing D.

A state-space graph represents D as one state with multiple incoming edges.

This distinction matters because graph search can use an explored set to
avoid repeatedly expanding the same state.
""")


# ============================================================
# SECTION 34
# PATH COST
# ============================================================

print("\n" + "=" * 80)
print("PATH COST")
print("=" * 80)

print("""
Path cost is the cumulative cost associated with a sequence of actions.

If:

    c(n, a, n') = cost of an individual transition

then:

    g(n) = sum of step costs along the path

Path cost allows an agent to distinguish between solutions that reach the
same goal but have different costs.

Examples of cost:

    distance
    time
    energy
    money
    risk
    computational resources

The meaning of "optimal" depends on the cost definition.
""")


path_costs = [4, 3, 7, 2]

total_cost = sum(path_costs)

print("Step costs:", path_costs)
print("Total path cost:", total_cost)


# ============================================================
# SECTION 35
# GOAL TEST
# ============================================================

print("\n" + "=" * 80)
print("GOAL TEST")
print("=" * 80)

print("""
A goal test determines whether a state satisfies the goal.

Example:

    Goal:
        Reach city Delhi.

Goal test:

    current_location == "Delhi"

A goal test can be simple or complex.

Examples:

    Check whether a puzzle is solved.
    Check whether all packages have been delivered.
    Check whether a robot reaches a target region.
    Check whether a schedule satisfies all constraints.

The goal test should describe the desired state rather than prescribing
one particular sequence of actions.
""")


def city_goal_test(city: str) -> bool:
    return city == "Delhi"


for city in ["Lucknow", "Kanpur", "Delhi"]:
    print(
        city,
        "->",
        city_goal_test(city)
    )


# ============================================================
# SECTION 36
# ACTION SEQUENCES
# ============================================================

print("\n" + "=" * 80)
print("ACTION SEQUENCES AND PLANS")
print("=" * 80)

print("""
A single action may not be enough to achieve a goal.

A plan is a sequence of actions.

Example:

    Initial state:
        Home

    Goal:
        Airport

    Plan:
        walk_to_station
        board_train
        exit_at_station
        take_taxi
        enter_airport

The sequence transforms the initial state into a goal state.

In deterministic environments, the resulting state may be predicted directly
if the transition model is known.

In stochastic environments, the same plan may lead to several possible
outcomes.
""")


plan = [
    "walk_to_station",
    "board_train",
    "travel",
    "exit_train",
    "take_taxi",
    "reach_airport",
]

print("Plan:")

for step_number, action_name in enumerate(plan, start=1):
    print(
        f"{step_number}. {action_name}"
    )


# ============================================================
# SECTION 37
# LEARNING AGENTS
# ============================================================

print("\n" + "=" * 80)
print("LEARNING AGENTS")
print("=" * 80)

print("""
A learning agent improves its behavior through experience.

A classical learning-agent architecture contains several conceptual
components:

    Performance element
        chooses actions.

    Learning element
        modifies the performance element.

    Critic
        evaluates behavior.

    Problem generator
        encourages useful exploration.

A simple feedback loop is:

    Environment
         |
         v
      Experience
         |
         v
       Critic
         |
         v
      Learning
         |
         v
Performance element
         |
         v
       Action
""")


class SimpleLearningAgent:

    def __init__(self):
        self.action_scores = {}
        self.learning_rate = 0.1

    def choose_action(self, actions: List[str]) -> str:

        if not actions:
            raise ValueError("No actions available.")

        unknown = [
            action
            for action in actions
            if action not in self.action_scores
        ]

        if unknown:
            return random.choice(unknown)

        return max(
            actions,
            key=lambda action: self.action_scores[action]
        )

    def learn(self, action: str, reward: float):

        old_value = self.action_scores.get(action, 0.0)

        new_value = (
            old_value
            + self.learning_rate
            * (reward - old_value)
        )

        self.action_scores[action] = new_value


learning_agent = SimpleLearningAgent()

random.seed(10)

for _ in range(20):

    action_name = learning_agent.choose_action(
        ["A", "B", "C"]
    )

    reward_map = {
        "A": 5,
        "B": 10,
        "C": 2
    }

    reward = reward_map[action_name]

    learning_agent.learn(
        action_name,
        reward
    )

print(
    "Learned action values:",
    learning_agent.action_scores
)


# ============================================================
# SECTION 38
# EXPLORATION VS EXPLOITATION
# ============================================================

print("\n" + "=" * 80)
print("EXPLORATION VS EXPLOITATION")
print("=" * 80)

print("""
A learning agent faces a fundamental decision:

    Exploration:
        try actions to obtain new information.

    Exploitation:
        choose actions already believed to be good.

If an agent always exploits, it may never discover better actions.

If it explores too much, it may repeatedly sacrifice performance.

This trade-off appears in:

    reinforcement learning
    recommendation systems
    online optimization
    adaptive control
    experimental decision making
""")


def epsilon_greedy(
    values: Dict[str, float],
    epsilon: float
) -> str:

    actions = list(values.keys())

    if random.random() < epsilon:
        return random.choice(actions)

    return max(
        actions,
        key=values.get
    )


random.seed(5)

action_values = {
    "A": 10,
    "B": 20,
    "C": 15
}

for _ in range(10):
    print(
        epsilon_greedy(
            action_values,
            epsilon=0.2
        )
    )


# ============================================================
# SECTION 39
# UNKNOWN ENVIRONMENTS
# ============================================================

print("\n" + "=" * 80)
print("KNOWN VS UNKNOWN ENVIRONMENTS")
print("=" * 80)

print("""
In a known environment, the agent knows the relevant rules of behavior.

Example:

    A chess program knows how legal chess moves work.

In an unknown environment, the agent may not know:

    - which actions are available,
    - what actions do,
    - how likely outcomes are,
    - which states are safe.

The agent may need to learn the environment model.

This introduces a distinction between:

    acting using a known model

and:

    learning the model through interaction.
""")


# ============================================================
# SECTION 40
# DYNAMIC ENVIRONMENTS
# ============================================================

print("\n" + "=" * 80)
print("STATIC VS DYNAMIC ENVIRONMENTS")
print("=" * 80)

print("""
A static environment remains unchanged while the agent deliberates.

A dynamic environment can change during deliberation.

Chess is relatively static during a player's turn.

Driving is highly dynamic.

A dynamic environment creates additional requirements:

    - rapid perception
    - continuous updating
    - prediction
    - real-time decision making
    - robustness to changing conditions

An action plan calculated five seconds ago may no longer be appropriate
in a rapidly changing environment.
""")


# ============================================================
# SECTION 41
# EPISODIC VS SEQUENTIAL
# ============================================================

print("\n" + "=" * 80)
print("EPISODIC VS SEQUENTIAL")
print("=" * 80)

print("""
In an episodic environment, each decision can be treated largely as a
separate episode.

Example:

    An image-classification system classifies one image at a time.

The action for one image generally does not determine the next image.

In a sequential environment, current actions affect future states.

Examples:

    chess
    driving
    navigation
    financial planning
    robotics

Sequential environments require the agent to consider long-term effects.
""")


# ============================================================
# SECTION 42
# DISCRETE VS CONTINUOUS
# ============================================================

print("\n" + "=" * 80)
print("DISCRETE VS CONTINUOUS")
print("=" * 80)

print("""
Discrete environments have countable states or actions.

Example:

    Chess:
        finite board positions and legal moves.

Continuous environments involve variables that can take values over
continuous ranges.

Example:

    vehicle speed:
        0.0 to 200.0 km/h

    steering angle:
        continuous range

Many real-world systems combine discrete and continuous components.

For example:

    discrete:
        brake / accelerate / turn

    continuous:
        exact brake pressure
        exact steering angle
        exact acceleration
""")


# ============================================================
# SECTION 43
# SINGLE-AGENT VS MULTI-AGENT
# ============================================================

print("\n" + "=" * 80)
print("SINGLE-AGENT VS MULTI-AGENT")
print("=" * 80)

print("""
In a single-agent environment, the primary uncertainty comes from
the environment itself.

In a multi-agent environment, other agents may actively choose actions.

Examples:

Single-agent:
    maze solving

Multi-agent:
    chess
    football
    autonomous traffic
    competitive markets

Other agents can be:

    cooperative
    competitive
    partially cooperative
    unpredictable

This changes the decision problem because the agent must account for
the behavior of other decision makers.
""")


# ============================================================
# SECTION 44
# RATIONAL DECISION UNDER UNCERTAINTY
# ============================================================

print("\n" + "=" * 80)
print("RATIONAL DECISION UNDER UNCERTAINTY")
print("=" * 80)

print("""
Suppose an agent has three possible actions.

Each action has several possible outcomes.

The agent can estimate:

    P(outcome | action)

and assign utility:

    U(outcome)

The decision criterion can then be:

    choose action maximizing expected utility.

This gives:

    a* = argmax_a EU(a)

where:

    a*       = chosen action
    EU(a)    = expected utility of action a

This mathematical idea forms a bridge between intelligent agents,
probabilistic reasoning and decision theory.
""")


@dataclass
class ProbabilisticOutcome:
    probability: float
    utility_value: float


@dataclass
class ProbabilisticAction:
    name: str
    outcomes: List[ProbabilisticOutcome]

    def expected_utility(self) -> float:
        return sum(
            outcome.probability
            * outcome.utility_value
            for outcome in self.outcomes
        )


probabilistic_actions = [
    ProbabilisticAction(
        "A",
        [
            ProbabilisticOutcome(0.7, 100),
            ProbabilisticOutcome(0.3, 10),
        ]
    ),
    ProbabilisticAction(
        "B",
        [
            ProbabilisticOutcome(0.4, 150),
            ProbabilisticOutcome(0.6, 50),
        ]
    ),
    ProbabilisticAction(
        "C",
        [
            ProbabilisticOutcome(0.9, 80),
            ProbabilisticOutcome(0.1, 60),
        ]
    ),
]

for action_option in probabilistic_actions:
    print(
        action_option.name,
        "->",
        action_option.expected_utility()
    )

best_action = max(
    probabilistic_actions,
    key=lambda x: x.expected_utility()
)

print(
    "Best expected-utility action:",
    best_action.name
)


# ============================================================
# SECTION 45
# DOMINANCE
# ============================================================

print("\n" + "=" * 80)
print("DOMINANCE IN DECISION MAKING")
print("=" * 80)

print("""
An action can be considered dominated when another action performs at least
as well under all relevant outcomes and strictly better under at least one.

Suppose:

    Action A:
        safety = 80
        cost = 20

    Action B:
        safety = 90
        cost = 20

If all else is equal, B dominates A because B provides greater safety
for the same cost.

Dominance can help eliminate clearly inferior choices before applying
more complex decision analysis.
""")


# ============================================================
# SECTION 46
# UTILITY AND PREFERENCE
# ============================================================

print("\n" + "=" * 80)
print("UTILITY AND PREFERENCES")
print("=" * 80)

print("""
Utility is a numerical representation of preference.

If:

    U(A) > U(B)

then the agent prefers A to B under that utility model.

Utility values themselves do not have to represent physical quantities.

For example:

    U(A) = 100
    U(B) = 50

does not necessarily mean A is "twice as good" in a literal physical sense.

The numbers provide an ordering or preference representation under
the chosen decision model.
""")


preferences = {
    "Outcome A": 90,
    "Outcome B": 70,
    "Outcome C": 40,
}

ranked = sorted(
    preferences.items(),
    key=lambda item: item[1],
    reverse=True
)

print("Preferences:")

for outcome_name, score in ranked:
    print(
        f"{outcome_name}: {score}"
    )


# ============================================================
# SECTION 47
# AGENT FUNCTION
# ============================================================

print("\n" + "=" * 80)
print("AGENT FUNCTION")
print("=" * 80)

print("""
An agent function conceptually maps percept sequences to actions:

    f:
        percept sequence -> action

For example:

    f([dirty]) = clean

    f([clean, obstacle]) = turn

    f([traffic_red, pedestrian]) = brake

The agent program implements this function on some computational platform.

This distinction is useful:

    Agent function:
        abstract mathematical description of behavior.

    Agent program:
        actual implementation.

    Architecture:
        hardware/software platform on which the program runs.
""")


def agent_function(percept_sequence: List[str]) -> str:

    if not percept_sequence:
        return "observe"

    current = percept_sequence[-1]

    rules = {
        "dirty": "clean",
        "obstacle": "turn",
        "danger": "escape",
        "goal": "stop",
    }

    return rules.get(current, "continue")


history = ["clear", "clear", "obstacle"]

print(
    "Percept sequence:",
    history
)

print(
    "Action:",
    agent_function(history)
)


# ============================================================
# SECTION 48
# INTERNAL STATE
# ============================================================

print("\n" + "=" * 80)
print("INTERNAL STATE")
print("=" * 80)

print("""
A purely reactive agent can act on current percepts.

An internal-state agent maintains information that is not directly contained
in the current percept.

For example:

Current percept:
    "door"

Internal state:
    door was previously open
    agent is carrying a package
    destination is room 4

This information allows richer decisions.

Internal state is especially important in partially observable environments.
""")


class StatefulAgent:

    def __init__(self):
        self.state = {
            "location": None,
            "carrying_package": False,
            "visited": set(),
        }

    def perceive(self, location: str):
        self.state["location"] = location
        self.state["visited"].add(location)

    def act(self):
        if not self.state["carrying_package"]:
            return "search_for_package"

        return "deliver_package"


stateful_agent = StatefulAgent()

stateful_agent.perceive("Warehouse")

print("Internal state:")
print(stateful_agent.state)

print(
    "Action:",
    stateful_agent.act()
)


# ============================================================
# SECTION 49
# PROBLEM-SOLVING AGENT CYCLE
# ============================================================

print("\n" + "=" * 80)
print("PROBLEM-SOLVING AGENT CYCLE")
print("=" * 80)

print("""
A classical problem-solving agent can be described through the following
cycle:

    1. Perceive
    2. Formulate a goal
    3. Formulate a problem
    4. Search for a solution
    5. Execute the solution
    6. Observe the result
    7. Repeat

The problem formulation converts a real-world objective into a formal
search problem.

The search algorithm then operates on the formal representation.
""")


@dataclass
class ProblemSolvingAgent:

    current_state: Any
    goal: Any

    def formulate_goal(self, goal: Any):
        self.goal = goal

    def formulate_problem(self):
        return {
            "initial_state": self.current_state,
            "goal": self.goal,
        }

    def execute(self, action: Any):
        self.current_state = action


problem_agent = ProblemSolvingAgent(
    current_state="Home",
    goal="Airport"
)

print(
    "Problem:",
    problem_agent.formulate_problem()
)


# ============================================================
# SECTION 50
# INTEGRATED GRID-WORLD AGENT
# ============================================================

print("\n" + "=" * 80)
print("INTEGRATED GRID-WORLD AGENT")
print("=" * 80)

print("""
The following example brings together:

    state
    action
    transition
    goal
    path cost
    heuristic
    search
    planning

The agent begins at a start state and searches for a path to the goal.
""")


class GridWorld:

    def __init__(
        self,
        rows: int,
        cols: int,
        obstacles: Optional[set] = None
    ):

        self.rows = rows
        self.cols = cols
        self.obstacles = obstacles or set()

    def successors(
        self,
        state: GridState
    ) -> List[Tuple[str, GridState]]:

        results = []

        for action_name in ACTIONS:

            next_state = apply_action(
                state,
                action_name,
                self.rows,
                self.cols
            )

            if next_state is None:
                continue

            if next_state in self.obstacles:
                continue

            results.append(
                (action_name, next_state)
            )

        return results


def grid_world_a_star(
    world: GridWorld,
    start: GridState,
    goal: GridState
) -> Optional[List[Tuple[str, GridState]]]:

    counter = 0

    start_node = Node(state=start)

    frontier = [
        (
            manhattan_distance(start, goal),
            counter,
            start_node
        )
    ]

    best_g = {start: 0}

    while frontier:

        _, _, node = heapq.heappop(frontier)

        if node.state == goal:

            path_nodes = []

            current = node

            while current.parent is not None:
                path_nodes.append(
                    (
                        current.action,
                        current.state
                    )
                )

                current = current.parent

            path_nodes.reverse()

            return path_nodes

        current_g = best_g[node.state]

        for action_name, next_state in world.successors(
            node.state
        ):

            new_g = current_g + 1

            if new_g < best_g.get(
                next_state,
                math.inf
            ):

                best_g[next_state] = new_g

                h = manhattan_distance(
                    next_state,
                    goal
                )

                f = new_g + h

                counter += 1

                child = Node(
                    state=next_state,
                    parent=node,
                    action=action_name,
                    path_cost=new_g,
                    depth=node.depth + 1
                )

                heapq.heappush(
                    frontier,
                    (f, counter, child)
                )

    return None


world = GridWorld(
    rows=6,
    cols=6,
    obstacles={
        GridState(1, 1),
        GridState(1, 2),
        GridState(2, 2),
        GridState(3, 2),
        GridState(4, 4),
    }
)

start = GridState(0, 0)
goal = GridState(5, 5)

solution = grid_world_a_star(
    world,
    start,
    goal
)

print("A* solution:")

if solution is None:
    print("No solution found.")

else:
    current = start

    print("Start:", current)

    for action_name, next_state in solution:
        print(
            f"Action: {action_name:5} "
            f"-> State: {next_state}"
        )

        current = next_state


# ============================================================
# SECTION 51
# STATE, ACTION, GOAL AND UTILITY TOGETHER
# ============================================================

print("\n" + "=" * 80)
print("STATE + ACTION + GOAL + UTILITY")
print("=" * 80)

print("""
These concepts should not be treated as isolated definitions.

Consider an autonomous delivery robot.

STATE:
    position
    battery
    package status
    current time
    known obstacles

ACTION:
    move
    recharge
    pick up
    drop off
    wait

GOAL:
    deliver package

UTILITY:
    high reward for successful and safe delivery
    lower utility for delays
    lower utility for energy consumption
    very low utility for unsafe behavior

The complete reasoning process can therefore be expressed as:

    perceive current state
              |
              v
       identify possible actions
              |
              v
       predict possible outcomes
              |
              v
       evaluate outcomes
              |
              v
       select rational action
              |
              v
        execute action
              |
              v
       observe new state
              |
              v
            repeat
""")


# ============================================================
# SECTION 52
# A MORE ADVANCED DECISION MODEL
# ============================================================

print("\n" + "=" * 80)
print("ADVANCED DECISION MODEL")
print("=" * 80)

print("""
Suppose an agent has actions:

    A1
    A2
    A3

Each action can result in several states.

The agent has:

    P(s' | s, a)

representing the probability of reaching state s' after taking action a
from state s.

Each state has utility:

    U(s')

The expected utility of an action is:

    EU(a | s)
        =
        Σ P(s' | s, a) U(s')

The rational decision is:

    a*
        =
        argmax_a EU(a | s)

This formulation is fundamental to rational decision making under
uncertainty.

For sequential problems, immediate utility alone may not be sufficient.
The agent may need to consider future states and cumulative utility.
""")


@dataclass
class TransitionOutcome:
    next_state: str
    probability: float


class DecisionEnvironment:

    def __init__(
        self,
        transition_model: Dict[
            Tuple[str, str],
            List[TransitionOutcome]
        ],
        utilities: Dict[str, float]
    ):

        self.transition_model = transition_model
        self.utilities = utilities

    def expected_utility(
        self,
        state: str,
        action_name: str
    ) -> float:

        outcomes = self.transition_model.get(
            (state, action_name),
            []
        )

        return sum(
            outcome.probability
            * self.utilities[outcome.next_state]
            for outcome in outcomes
        )

    def best_action(
        self,
        state: str,
        actions: List[str]
    ) -> str:

        return max(
            actions,
            key=lambda action_name:
            self.expected_utility(
                state,
                action_name
            )
        )


decision_environment = DecisionEnvironment(
    transition_model={
        ("S", "A"): [
            TransitionOutcome("G1", 0.8),
            TransitionOutcome("G2", 0.2),
        ],
        ("S", "B"): [
            TransitionOutcome("G1", 0.4),
            TransitionOutcome("G2", 0.6),
        ],
        ("S", "C"): [
            TransitionOutcome("G1", 0.9),
            TransitionOutcome("G3", 0.1),
        ],
    },
    utilities={
        "G1": 100,
        "G2": 20,
        "G3": 50,
    }
)

for action_name in ["A", "B", "C"]:

    print(
        action_name,
        "->",
        decision_environment.expected_utility(
            "S",
            action_name
        )
    )

print(
    "Rational action:",
    decision_environment.best_action(
        "S",
        ["A", "B", "C"]
    )
)


# ============================================================
# SECTION 53
# CUMULATIVE UTILITY
# ============================================================

print("\n" + "=" * 80)
print("CUMULATIVE UTILITY")
print("=" * 80)

print("""
In sequential environments, an agent often cares about a sequence of
rewards rather than a single immediate outcome.

A simple cumulative utility model is:

    U_total =
        U_1 + U_2 + U_3 + ... + U_n

A discounted model can be:

    U_total =
        R_1
        + γR_2
        + γ²R_3
        + ...

where:

    γ = discount factor

with:

    0 <= γ <= 1

A smaller γ makes future rewards less influential.

A larger γ makes future rewards relatively more important.
""")


def discounted_return(
    rewards: List[float],
    gamma: float
) -> float:

    total = 0.0

    for t, reward in enumerate(rewards):
        total += (gamma ** t) * reward

    return total


rewards = [10, 20, 30, 40]

print(
    "Undiscounted return:",
    discounted_return(rewards, 1.0)
)

print(
    "Discounted return:",
    discounted_return(rewards, 0.9)
)


# ============================================================
# SECTION 54
# UTILITY AND GOAL CONFLICT
# ============================================================

print("\n" + "=" * 80)
print("CONFLICTING GOALS")
print("=" * 80)

print("""
An agent can have multiple objectives.

Example:

    Goal 1:
        minimize travel time

    Goal 2:
        maximize safety

These goals may conflict.

A faster route may be less safe.

A safer route may take longer.

A binary goal representation cannot easily express the preference among
multiple successful outcomes.

A utility function provides a way to encode the trade-off.
""")


goal_tradeoffs = {
    "Fast Route": {
        "time": 95,
        "safety": 65
    },
    "Balanced Route": {
        "time": 80,
        "safety": 85
    },
    "Safe Route": {
        "time": 55,
        "safety": 98
    }
}

for route_name, values in goal_tradeoffs.items():

    score = (
        0.4 * values["time"]
        + 0.6 * values["safety"]
    )

    print(
        route_name,
        "utility =",
        score
    )


# ============================================================
# SECTION 55
# INFORMATION VALUE
# ============================================================

print("\n" + "=" * 80)
print("INFORMATION AND DECISION MAKING")
print("=" * 80)

print("""
An intelligent agent may sometimes benefit from gathering additional
information before acting.

Example:

A robot must choose between two routes.

Without additional information:

    Route A appears slightly better.

The robot could first inspect the environment.

The inspection itself has a cost:

    time
    energy
    computation
    risk

Therefore information gathering is itself a decision problem.

An intelligent system must balance:

    value of information

against:

    cost of obtaining information.
""")


@dataclass
class InformationAction:
    name: str
    information_value: float
    information_cost: float

    def net_value(self) -> float:
        return (
            self.information_value
            - self.information_cost
        )


information_actions = [
    InformationAction(
        "Inspect",
        information_value=25,
        information_cost=8
    ),
    InformationAction(
        "Act Immediately",
        information_value=0,
        information_cost=0
    ),
]

for option in information_actions:
    print(
        option.name,
        "-> net value:",
        option.net_value()
    )


# ============================================================
# SECTION 56
# AGENT DECISION PIPELINE
# ============================================================

print("\n" + "=" * 80)
print("COMPLETE AGENT DECISION PIPELINE")
print("=" * 80)

print("""
A generalized intelligent-agent decision process can be represented as:

    1. PERCEPTION
       Receive information.

    2. STATE ESTIMATION
       Determine the relevant current state or belief state.

    3. GOAL IDENTIFICATION
       Determine desired conditions.

    4. ACTION GENERATION
       Identify feasible actions.

    5. PREDICTION
       Estimate possible consequences.

    6. EVALUATION
       Calculate cost, utility or expected utility.

    7. DECISION
       Select an action according to the decision criterion.

    8. EXECUTION
       Perform the selected action.

    9. FEEDBACK
       Observe the result.

   10. UPDATE
       Revise internal state, model or knowledge.

The exact implementation differs among agent architectures.
""")


@dataclass
class IntelligentAgentPipeline:

    state: Any
    goal: Any
    utility_function: Callable[[Any], float]

    def perceive(self, percept: Any):
        self.state = percept

    def evaluate_actions(
        self,
        actions: Dict[str, Any]
    ) -> Dict[str, float]:

        return {
            action_name: self.utility_function(result)
            for action_name, result in actions.items()
        }

    def decide(
        self,
        actions: Dict[str, Any]
    ) -> str:

        utilities = self.evaluate_actions(actions)

        return max(
            utilities,
            key=utilities.get
        )


pipeline_agent = IntelligentAgentPipeline(
    state="S0",
    goal="G",
    utility_function=lambda x: {
        "result_A": 60,
        "result_B": 90,
        "result_C": 75
    }.get(x, 0)
)

candidate_actions = {
    "action_A": "result_A",
    "action_B": "result_B",
    "action_C": "result_C",
}

print(
    "Selected action:",
    pipeline_agent.decide(
        candidate_actions
    )
)


# ============================================================
# SECTION 57
# IMPORTANT CONCEPTUAL DISTINCTIONS
# ============================================================

print("\n" + "=" * 80)
print("IMPORTANT CONCEPTUAL DISTINCTIONS")
print("=" * 80)

print("""
STATE
-----
The current condition relevant to decision making.

ACTION
------
A choice available to the agent.

TRANSITION
----------
The change caused by an action.

GOAL
----
A desired condition.

PATH
----
A sequence of states or actions.

PATH COST
---------
The cumulative cost of reaching a state.

UTILITY
-------
A numerical representation of preference.

RATIONALITY
-----------
Choosing the action expected to maximize performance according to the
available information and decision model.

HEURISTIC
---------
An estimate used to guide search.

SEARCH
------
Systematic exploration of a state space to find a solution.

POLICY
------
A mapping from states or observations to actions.

MODEL
-----
A representation of how the environment behaves.

BELIEF STATE
------------
A representation of uncertainty about the actual state.

PERFORMANCE MEASURE
-------------------
The criterion used to evaluate how well the agent behaves externally.
""")


# ============================================================
# SECTION 58
# POLICY
# ============================================================

print("\n" + "=" * 80)
print("POLICY")
print("=" * 80)

print("""
A policy specifies what action an agent should take given a state or
observation.

Conceptually:

    π(s) = action

Example:

    π(A) = RIGHT
    π(B) = DOWN
    π(C) = LEFT

A policy differs from a single action.

An action is one decision.

A policy is a rule or mapping that can determine actions across many states.

This distinction becomes central in reinforcement learning and sequential
decision making.
""")


policy = {
    "A": "RIGHT",
    "B": "DOWN",
    "C": "LEFT",
    "D": "UP",
}


def get_action(policy: Dict[str, str], state: str) -> str:
    return policy.get(state, "NO_ACTION")


for state_name in policy:
    print(
        f"State {state_name} -> Action {get_action(policy, state_name)}"
    )


# ============================================================
# SECTION 59
# SEARCH VS DECISION THEORY
# ============================================================

print("\n" + "=" * 80)
print("SEARCH VS DECISION MAKING")
print("=" * 80)

print("""
Classical search often assumes that the agent can construct a solution
by exploring possible sequences of actions.

Decision theory focuses on selecting actions when outcomes can have
different probabilities and utilities.

Search asks:

    "Which sequence leads to a goal?"

Decision theory asks:

    "Which action or strategy is most desirable given uncertainty
     and preferences?"

In real systems, these ideas can be combined.

A navigation system may:

    search for routes

while also considering:

    traffic probability
    travel time
    fuel
    safety
    uncertainty

The resulting system is not merely finding any path. It is making a
decision using a model of preferences and uncertainty.
""")


# ============================================================
# SECTION 60
# COMPLETE MINI EXAMPLE
# ============================================================

print("\n" + "=" * 80)
print("COMPLETE MINI EXAMPLE: DELIVERY ROBOT")
print("=" * 80)

print("""
Task:

    A robot must deliver a package from A to G.

Environment:

    grid with obstacles

State:

    robot location

Actions:

    UP, DOWN, LEFT, RIGHT

Goal:

    reach G

Cost:

    one unit per movement

Heuristic:

    Manhattan distance

Decision method:

    A* search

This is a simple deterministic version of intelligent problem solving.
""")


delivery_world = GridWorld(
    rows=7,
    cols=7,
    obstacles={
        GridState(1, 1),
        GridState(1, 2),
        GridState(2, 2),
        GridState(3, 2),
        GridState(4, 2),
        GridState(4, 3),
        GridState(4, 4),
        GridState(5, 4),
    }
)

delivery_start = GridState(0, 0)
delivery_goal = GridState(6, 6)

delivery_solution = grid_world_a_star(
    delivery_world,
    delivery_start,
    delivery_goal
)

if delivery_solution:

    print("\nDelivery plan:")

    current_state = delivery_start

    for step_number, (action_name, next_state) in enumerate(
        delivery_solution,
        start=1
    ):

        print(
            f"{step_number:2}. "
            f"{current_state} "
            f"--{action_name}--> "
            f"{next_state}"
        )

        current_state = next_state

else:
    print("No delivery path found.")


# ============================================================
# SECTION 61
# FINAL CONCEPTUAL MODEL
# ============================================================

print("\n" + "=" * 80)
print("CONCEPTUAL MODEL")
print("=" * 80)

print("""
The central structure of AI problem solving can be understood as:

                    ENVIRONMENT
                         |
                         v
                      PERCEPT
                         |
                         v
                       AGENT
                         |
             +-----------+-----------+
             |                       |
             v                       v
        CURRENT STATE              GOAL
             |                       |
             +-----------+-----------+
                         |
                         v
                    ACTIONS
                         |
                         v
                   PREDICT RESULTS
                         |
                         v
                 COST / UTILITY
                         |
                         v
                 RATIONAL DECISION
                         |
                         v
                       ACTION
                         |
                         v
                    ENVIRONMENT
                         |
                         +------> NEW PERCEPT
                                  |
                                  v
                               REPEAT

The fundamental relationships are:

    State
        describes where the agent believes it is.

    Action
        changes the state.

    Goal
        describes a desired condition.

    Search
        finds sequences of actions that can achieve goals.

    Cost
        measures the resources required by a solution.

    Utility
        measures preference among possible outcomes.

    Rationality
        determines which action is expected to perform best according
        to the available information and the agent's objectives.

    Environment
        determines what the agent can perceive and how actions affect
        the world.

    Agent architecture
        determines how perception, internal state, goals, utility,
        learning and action selection are implemented.

Together these concepts provide the basic formal vocabulary for
understanding intelligent agents and AI problem solving.
""")

print("=" * 80)
print("END OF AI PROBLEM SOLVING STUDY PROGRAM")
print("=" * 80)
