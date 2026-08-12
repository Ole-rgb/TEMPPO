"""LLM-proposed subgoals as a PPO exploration bonus.

The shaped reward is folded into the extrinsic stream before GAE, exactly like RND:
    r_t = r_ext_t + beta_llm * r_subgoal_t

An LLM is shown the full map **class plan** once and returns an ordered list of subgoals
("pick up the yellow key", "open the yellow door", "go to the goal"). During
training the agent collects `subgoal_bonus` the first time it satisfies the next
outstanding subgoal in that list. No API call ever happens inside the training loop.

WHY A PLAN CLASS AND NOT A MAP HASH
MiniGrid regenerates the layout every episode, so hashing the rendered map means one cache
miss per episode -- ~12.5k per MultiRoom run, ~225k across a sweep. `plan_key` instead
canonicalises the map under two transformations that cannot change the plan: room-graph and
colour permutation. Both are SYMMETRIES of the environment, and solely used for environment
caching --  the LLM only receives the full rendered map. Measured over 6k episodes; the fraction
of later episodes whose class was already seen (see `audit_env`):

    env                     plan classes   coverage   ms/key
    DoorKey-5x5/8x8/16x16              1       100%      0.2
    UnlockPickup                       2       100%      0.3
    LockedRoom                         3       100%      0.8
    MultiRoom-N4-S5-v1                 7       100%      1.1
    MultiRoom-N6                      67       100%      1.2
    ObstructedMaze-Full              312       100%      0.8
    KeyCorridorS3R3                5,260        13%      0.6   <- unbounded

A small class count is necessary but not sufficient: ObstructedMaze-Full keys cleanly yet
hides its keys inside boxes, which `pick_up / open_door / go_to_goal` cannot express.
Cacheable is about cost; solvable is about the schema.

Everything is gated on `cfg.bonus.beta_llm > 0`; `cfg.llm.*` holds hyperparameters that
stay FIXED across conditions.
"""

import re
from pathlib import Path
from typing import Literal

import networkx as nx
import numpy as np
from pydantic import BaseModel

# --- map rendering ----------------------------------------------------------------

OBJ_CHAR = {
    "wall": "W",
    "floor": "F",
    "door": "D",
    "key": "K",
    "ball": "A",
    "box": "B",
    "goal": "G",
    "lava": "V",
}
COLOR_CHAR = {"red": "R", "green": "G", "blue": "B", "purple": "P", "yellow": "Y", "grey": "E"}
AGENT_CHAR = {0: ">", 1: "V", 2: "<", 3: "^"}

COLORS = Literal["red", "green", "blue", "purple", "yellow", "grey"]
OBJECTS = Literal["key", "ball", "box"]
ACTIONS = Literal["pick_up", "open_door", "go_to_goal"]


def render_map(base) -> str:
    """The full grid as 2-char cells: object character then colour character.

    Doors carry their state instead of a bare 'D': 'L<c>' locked, 'D<c>' closed,
    '__' already open. The agent is a doubled arrow for the way it faces.
    """
    grid = base.grid
    rows = []
    for j in range(grid.height):
        row = []
        for i in range(grid.width):
            if (i, j) == tuple(base.agent_pos):
                row.append(AGENT_CHAR[base.agent_dir] * 2)
                continue
            obj = grid.get(i, j)
            if obj is None:
                row.append("  ")
            elif obj.type == "door":
                if obj.is_open:
                    row.append("__")
                else:
                    row.append(("L" if obj.is_locked else "D") + COLOR_CHAR[obj.color])
            else:
                row.append(OBJ_CHAR[obj.type] + COLOR_CHAR[obj.color])
        rows.append("".join(row))
    return "\n".join(rows)


# --- plan classes -----------------------------------------------------------------

WALKABLE = {"goal", "key", "ball", "box", "lava", "floor"}


def room_graph(base) -> nx.Graph:
    """The map as rooms and doors, flood-filled straight off the grid.

    Rooms are 4-connected components of non-wall, non-door cells; doors are the
    separators between them. Nodes carry the room's contents and whether the agent starts
    there, edges carry the door's colour and locked flag.

    Deliberately knows nothing about MiniGrid env families -- no `base.rooms`, no notion
    of which doors "matter". It records what is there; deciding what to do about it is the
    LLM's job.
    """
    grid, w, h = base.grid, base.grid.width, base.grid.height

    def walkable(i, j):
        obj = grid.get(i, j)
        return obj is None or obj.type in WALKABLE

    room_of, n_rooms = {}, 0
    for j in range(h):
        for i in range(w):
            if not walkable(i, j) or (i, j) in room_of:
                continue
            n_rooms += 1
            stack, room_of[(i, j)] = [(i, j)], n_rooms
            while stack:
                x, y = stack.pop()
                for p in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if 0 <= p[0] < w and 0 <= p[1] < h and p not in room_of and walkable(*p):
                        room_of[p] = n_rooms
                        stack.append(p)

    graph = nx.Graph()
    objs = {r: [] for r in range(1, n_rooms + 1)}
    for (i, j), r in room_of.items():
        obj = grid.get(i, j)
        if obj is not None and obj.type != "floor":
            objs[r].append((obj.type, obj.color))
    agent_room = room_of.get(tuple(base.agent_pos))
    for r in range(1, n_rooms + 1):
        graph.add_node(r, objs=tuple(sorted(objs[r])), agent=(r == agent_room))

    for j in range(h):
        for i in range(w):
            obj = grid.get(i, j)
            if obj is None or obj.type != "door":
                continue
            sides = sorted(
                {
                    room_of[p]
                    for p in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1))
                    if p in room_of
                }
            )
            if len(sides) == 2:
                graph.add_edge(*sides, color=obj.color, locked=bool(obj.is_locked))
            elif len(sides) == 1:
                # A door onto a dead end. Give it a stub node so it still shows up.
                stub = f"stub{i},{j}"
                graph.add_node(stub, objs=(), agent=False)
                graph.add_edge(sides[0], stub, color=obj.color, locked=bool(obj.is_locked))

    # Keeping everything with no doors and no obstacles out (empty space around and dead rooms)
    if agent_room is not None:
        graph = graph.subgraph(nx.node_connected_component(graph, agent_room)).copy()

    # Renumber rooms 1..n
    return nx.convert_node_labels_to_integers(graph, first_label=1, ordering="sorted")


def _hash_graph(graph, recolour) -> str:
    """WL hash of `graph` with colours mapped through `recolour` (identity if None)."""
    sub = (lambda c: c) if recolour is None else (lambda c: recolour.get(c, c))
    labelled = nx.Graph()
    for n, d in graph.nodes(data=True):
        tag = ",".join(f"{t}:{sub(c)}" for t, c in d["objs"])
        labelled.add_node(n, label=tag + ("|agent" if d["agent"] else ""))
    for a, b, d in graph.edges(data=True):
        labelled.add_edge(a, b, label=sub(d["color"]) + ("!" if d["locked"] else ""))
    return nx.weisfeiler_lehman_graph_hash(
        labelled, node_attr="label", edge_attr="label", iterations=3
    )


def _colour_roles(base, graph) -> dict:
    """Map each colour to a role index, canonically and without brute force.

    Colours are ordered by WHERE they appear in the colour-blind graph, so the ordering is
    itself invariant to which colours were drawn. Colours whose structural signatures tie
    are interchangeable, so falling back to alphabetical for them is safe: the two
    labellings are isomorphic and hash identically.

    Brute-forcing all permutations gives the same answer at ~300ms per layout instead of
    ~1ms, which is the difference between usable and not at 12.5k episodes per run.
    """
    blind = nx.Graph()
    for n, d in graph.nodes(data=True):
        blind.add_node(
            n, label=",".join(t for t, _ in d["objs"]) + ("|agent" if d["agent"] else "")
        )
    for a, b, d in graph.edges(data=True):
        blind.add_edge(a, b, label="!" if d["locked"] else "")
    node_hash = nx.weisfeiler_lehman_subgraph_hashes(
        blind, node_attr="label", edge_attr="label", iterations=3, include_initial_labels=True
    )
    sig = {}
    for n, d in graph.nodes(data=True):
        for t, c in d["objs"]:
            sig.setdefault(c, []).append(("obj", t, tuple(node_hash[n])))
    for a, b, d in graph.edges(data=True):
        ends = tuple(sorted((tuple(node_hash[a]), tuple(node_hash[b]))))
        sig.setdefault(d["color"], []).append(("door", d["locked"], ends))
    for c in COLOR_CHAR:
        if re.search(rf"\b{c}\b", base.mission):
            sig.setdefault(c, []).append(("mission", "", ()))

    order = sorted(sig, key=lambda c: (sorted(map(repr, sig[c])), c))
    return {c: f"c{k}" for k, c in enumerate(order)}


def plan_key_and_roles(base) -> tuple[str, dict]:
    """Canonical form of the map, plus the colour->role map that produced it.

    Exactly two transformations are removed:

      * room-graph isomorphism -- absolute positions and room numbering are irrelevant,
        only which room holds what and which doors join them;
      * colour permutation -- recolouring consistently everywhere (doors, keys, objects,
        mission) maps a plan to that same plan with colours relabelled.

    Nothing here decides which doors are subgoals or in what order to open them; the LLM
    still gets the full rendered map and has to work that out. Two layouts sharing a key
    are structurally the same problem, which is what makes the cache finite -- run
    `audit_env` to check the space actually closes before paying for it.

    Note: Conceptually an env containing grey-yellow-grey and red-blue-red are alike, so
    we can avoid dublicates by changing to color1-color2-color1 (which is identical for both)
    See `subgoals_to_roles` / `subgoals_from_roles`.
    """
    graph = room_graph(base)
    roles = _colour_roles(base, graph)
    mission = re.sub(rf"\b({'|'.join(COLOR_CHAR)})\b", lambda m: roles[m.group(1)], base.mission)
    return f"{mission}|{_hash_graph(graph, roles)}", roles


def plan_key(base) -> str:
    """Just the key -- see `plan_key_and_roles`."""
    return plan_key_and_roles(base)[0]


def audit_env(env_id: str, episodes: int = 10_000, budget: int = 500) -> dict:
    """Is this env cacheable? Run BEFORE spending API budget on a new environment.

    One failure mode now that ordering is never asserted: the plan classes keep growing,
    so the cache is unbounded. That happens when the layout's ROOM TOPOLOGY is redrawn
    each episode rather than just its colours -- KeyCorridor changes how many doors join
    which rooms every reset, so a faithful key cannot collapse it.

    `coverage` is the number that predicts training directly: buy plans for every class
    seen in the first half, then measure the fraction of SECOND-half episodes that hit one.
    That is the fraction of episodes which would actually receive their subgoal bonus, and
    it separates a slow converging tail from real divergence far better than asking
    whether any new class appeared at all:

        MultiRoom-N4-S5-v1   18 classes at 20k episodes, coverage ~1.00   fine
        MultiRoom-N6        205 classes at 20k episodes, coverage ~0.99   fine
        KeyCorridorS3R3  14,706 classes at 20k episodes, coverage ~0.45   unusable

    A miss is never fatal -- `SubgoalTracker` just pays no bonus that episode and counts
    it -- so `coverage` is a quality number, not a correctness one.
    """
    # Local: training never needs this helper. `minigrid` is what registers the env ids.
    import gymnasium as gym
    import minigrid  # noqa: F401

    base = gym.make(env_id).unwrapped
    base.reset(seed=0)
    half = episodes // 2

    first, hits, total = set(), 0, 0
    seen = set()
    for k in range(episodes):
        key = plan_key(base)
        seen.add(key)
        if k < half:
            first.add(key)
        else:
            total += 1
            hits += key in first
        base.reset()

    coverage = hits / total if total else 1.0
    return {
        "env_id": env_id,
        "plan_classes": len(seen),
        "coverage": coverage,
        "n_rooms": room_graph(base).number_of_nodes(),
        "cacheable": len(seen) <= budget and coverage >= 0.95,
    }


# --- schema + cache ---------------------------------------------------------------


class Subgoal(BaseModel):
    action: ACTIONS
    color: COLORS | None = None
    object: OBJECTS | None = None


class SubgoalList(BaseModel):
    subgoals: list[Subgoal]


class PlanStep(BaseModel):
    """A subgoal in ROLE space: `color` is 'c0', 'c1', ... rather than 'red', 'blue'.

    This is what gets cached. A plan class holds grey-yellow-grey and red-blue-red alike,
    so storing concrete colours would hand every member but one a plan naming doors that
    do not exist on its map.
    """

    action: ACTIONS
    color: str | None = None
    object: OBJECTS | None = None


class CacheElement(BaseModel):
    """One plan class: the key, a representative map, and the plan in ROLE space."""

    key: str
    mission: str
    environment: str
    subgoals: list[PlanStep]


def subgoals_to_roles(subgoals: list[Subgoal], roles: dict) -> list[PlanStep]:
    """Concrete plan -> role space, using the roles of the layout it was written for."""
    return [
        PlanStep(action=s.action, color=roles.get(s.color) if s.color else None, object=s.object)
        for s in subgoals
    ]


def subgoals_from_roles(steps: list[PlanStep], roles: dict) -> list[Subgoal] | None:
    """Role plan -> the concrete colours of THIS layout. None if a role is unbindable.

    `roles` maps colour->role, so binding inverts it. Every layout in a class carries the
    same set of roles by construction, so the inverse is total; returning None guards the
    case where a stale cache predates a change to `plan_key`.
    """
    back = {role: colour for colour, role in roles.items()}
    out = []
    for s in steps:
        if s.color is None:
            out.append(Subgoal(action=s.action, object=s.object))
            continue
        colour = back.get(s.color)
        if colour is None:
            return None
        out.append(Subgoal(action=s.action, color=colour, object=s.object))
    return out


class SubgoalCache:
    """Append-only JSONL of plan classes, loaded once into memory.

    Append-only so a precompute run that dies partway keeps everything it paid for.
    Plans live in role space here; `get` binds them to a concrete layout.
    """

    def __init__(self, path):
        self.path = Path(path)
        self._entries: dict[str, list[PlanStep]] = {}
        if self.path.exists():
            with open(self.path) as f:
                for line in f:
                    if line.strip():
                        rec = CacheElement.model_validate_json(line)
                        self._entries[rec.key] = rec.subgoals

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, key: str, roles: dict) -> list[Subgoal] | None:
        """The plan for `key`, with colours bound to the layout `roles` came from."""
        steps = self._entries.get(key)
        return None if steps is None else subgoals_from_roles(steps, roles)

    def add(self, key: str, mission: str, environment: str, subgoals, roles: dict) -> None:
        """Store a plan. `subgoals` may be concrete (from the LLM) or already role-space."""
        if key in self._entries:
            return
        steps = (
            subgoals
            if subgoals and isinstance(subgoals[0], PlanStep)
            else subgoals_to_roles(subgoals, roles)
        )
        element = CacheElement(key=key, mission=mission, environment=environment, subgoals=steps)
        self._entries[key] = element.subgoals
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(element.model_dump_json() + "\n")


# --- the LLM call -----------------------------------------------------------------

SYSTEM = """You are a subgoal planner for an agent in a MiniGrid gridworld.
You are given the FULL map of the environment and the agent's mission.
Output the ordered list of subgoals the agent must complete to accomplish the mission.

HOW TO READ THE MAP
The map is printed row by row, from the top row down. Each cell is exactly 2 characters:
the first is the OBJECT, the second is its COLOR.

Objects (1st character):
   W = wall
   D = door
   K = key
   A = ball
   B = box
   G = goal
   V = lava
   F = floor
   '  ' (two spaces) = empty floor (walkable)

Colors (2nd character):
   R = red, G = green, B = blue, P = purple, Y = yellow, E = grey

So 'WE' = grey wall, 'GG' = green goal, 'VR' = lava (lava is always red),
'KY' = yellow key, 'AB' = blue ball.

Doors are special:
   L<color>  = a LOCKED door (e.g. 'LY' = locked yellow door)
   D<color>  = a closed but UNLOCKED door (e.g. 'DB' = closed blue door)
   __        = an already OPEN door (its color is not shown, and it needs no action)

The agent is two identical arrows showing the way it faces:
   >> east,  VV south,  << west,  ^^ north
Note: 'VV' (two arrows) is the agent facing south; lava is 'VR' (V plus a color).
The agent's cell shows the agent, not whatever it is standing on.

Coordinates: columns count left->right from 0 (i); rows count top->bottom from 0 (j);
a cell is (i, j). Use coordinates only to reason about what exists and what blocks what.

YOUR TASK
From the mission and the map, decide which subgoals the agent must achieve and in what order.
A LOCKED door ('L<color>') requires first picking up the matching-color key.
A closed unlocked door ('D<color>') only needs to be opened - no key.
An open door ('__') needs no subgoal at all.
To reach a room you must open the door(s) leading into it. Only propose subgoals for objects
that actually appear on the map or are named in the mission.

SUBGOAL FIELDS
Each subgoal has:
  action: one of 'pick_up', 'open_door', 'go_to_goal'
  color:  required for 'pick_up' and 'open_door' (red, green, blue, purple, yellow, grey)
  object: required for 'pick_up' (key, ball, box)
'go_to_goal' takes no color or object. List subgoals in the order the agent should perform them."""


def make_prompt(mission: str, environment: str) -> str:
    return f"Mission: {mission}\nMap:\n{environment}"


def fetch_subgoals(client, model: str, prompt: str) -> list[Subgoal]:
    """One planning call. `anthropic` is imported by the caller, never by training."""
    resp = client.messages.parse(
        model=model,
        max_tokens=1024,
        system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": prompt}],
        output_format=SubgoalList,
    )
    return resp.parsed_output.subgoals


# --- detection --------------------------------------------------------------------


class SubgoalTracker:
    """Per-env progress along a cached plan, and the bonus for advancing it.

    Detection reads the underlying MiniGridEnv rather than the observation, because a
    7x7 egocentric view cannot see whether a door across the map is open.

    Autoreset timing matters here. Gymnasium's SyncVectorEnv runs AutoresetMode.NEXT_STEP:
    when `done` comes back True the base env still holds the finished episode's final
    state, and the reset lands on the FOLLOWING step. So the plan is refreshed one step
    after `done`, and that step is scored as zero.
    """

    def __init__(self, envs, cache: SubgoalCache, subgoal_bonus: float):
        self.bases = [e.unwrapped for e in envs.envs]
        self.cache = cache
        self.subgoal_bonus = float(subgoal_bonus)
        self.num_envs = len(self.bases)

        self.plans: list[list[Subgoal]] = [[] for _ in range(self.num_envs)]
        self.idx = np.zeros(self.num_envs, dtype=np.int64)
        self._credited_doors: list[set] = [set() for _ in range(self.num_envs)]
        # True where the env is about to be reset by the next envs.step().
        self._pending_reset = np.zeros(self.num_envs, dtype=bool)

        self.misses = 0  # plan classes absent from the cache -- logged, never fatal
        self.completed = 0  # subgoals credited, across all envs

        # `grid` exists from __init__ but is EMPTY until reset, so keying an unreset env
        # yields a degenerate key that misses every time. `agent_pos` is the honest test.
        if any(b.agent_pos is None for b in self.bases):
            raise RuntimeError(
                "SubgoalTracker needs a generated layout, so build it AFTER envs.reset() "
                "-- MiniGrid only fills the grid on reset."
            )
        for i in range(self.num_envs):
            self._replan(i)

    def _replan(self, i: int) -> None:
        # The roles come from THIS layout, so a plan cached as "open c0 then c1 then c0"
        # binds to whatever colours this episode actually drew.
        key, roles = plan_key_and_roles(self.bases[i])
        plan = self.cache.get(key, roles)
        if plan is None:
            self.misses += 1
            plan = []
        self.plans[i] = plan
        self.idx[i] = 0
        self._credited_doors[i].clear()

    def _achieved(self, i: int, sg: Subgoal, terminated: bool) -> bool:
        base = self.bases[i]

        if sg.action == "go_to_goal":
            # TODO might just be inflating extr. reward
            # MiniGrid terminates on the goal (and on lava, which these envs lack).
            return terminated

        if sg.action == "pick_up":
            held = base.carrying
            return held is not None and held.type == sg.object and held.color == sg.color

        if sg.action == "open_door":
            # A colour can repeat in a plan (MultiRoom allows blue-green-blue), so credit
            # a door only once, by position.
            for j in range(base.grid.height):
                for x in range(base.grid.width):
                    obj = base.grid.get(x, j)
                    if (
                        obj is not None
                        and obj.type == "door"
                        and obj.color == sg.color
                        and obj.is_open
                        and (x, j) not in self._credited_doors[i]
                    ):
                        self._credited_doors[i].add((x, j))
                        return True
            return False

        return False

    def step(self, dones, terminateds) -> np.ndarray:
        """Bonus per env for the transition that just finished. Call after `envs.step`.

        `dones` is terminated OR truncated; `terminateds` excludes timeouts.
        """
        dones = np.asarray(dones, dtype=bool).reshape(-1)
        terminateds = np.asarray(terminateds, dtype=bool).reshape(-1)
        bonus = np.zeros(self.num_envs, dtype=np.float32)

        for i in range(self.num_envs):
            if self._pending_reset[i]:
                # This step reset the env; the map is new and nothing was achieved on it.
                self._replan(i)
                continue
            plan = self.plans[i]
            while self.idx[i] < len(plan) and self._achieved(i, plan[self.idx[i]], terminateds[i]):
                self.idx[i] += 1
                self.completed += 1
                bonus[i] += self.subgoal_bonus

        self._pending_reset = dones.copy()
        return bonus


class LLMBonus:
    """Top-level handle, mirroring `RND`'s role in the training loop.

    Reads only the cache -- a missing plan class costs that episode its bonus and bumps
    `tracker.misses`, it never blocks training on an API call. Populate the cache with
    the offline precompute script first.
    """

    def __init__(self, cfg, envs):
        self.cache = SubgoalCache(cfg.llm.cache_path)
        self.tracker = SubgoalTracker(envs, self.cache, cfg.llm.subgoal_bonus)

    def step(self, dones, terminateds) -> np.ndarray:
        return self.tracker.step(dones, terminateds)

    @property
    def stats(self) -> dict:
        return {
            "llm/plan_classes_cached": len(self.cache),
            "llm/plan_cache_misses": self.tracker.misses,
            "llm/subgoals_completed": self.tracker.completed,
        }
