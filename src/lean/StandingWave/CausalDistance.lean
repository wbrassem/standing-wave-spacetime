/-
Copyright (c) 2026 Wayne Brassem. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Wayne Brassem
-/

import StandingWave.Locality

/-!
# Structural Reachability, Hop Distance, and Bounded Propagation

This module investigates the mathematical consequences of
relational locality in a fixed oscillator network.

Building on the oscillator configurations, admissible local
interactions, and finite event histories defined in
`StandingWave.Locality`, it develops structural connectivity,
a derived extended hop metric, and finite-event bounds on
the propagation of configuration differences.

The central objective is to determine which restrictions
on interaction and influence follow from the network
structure without assuming physical spacetime, geometric
distance, elapsed time, or a propagation speed.

## Foundational Assumptions

An oscillator network consists of vertices equipped with
a symmetric adjacency relation.

Permitted pairwise interaction events require adjacent,
distinct participating vertices.

An interaction reads and updates only its two participating
oscillator states. Every other vertex remains unchanged.

The same interaction applied to identical input states
produces identical outputs.

The network is fixed throughout the present development,
and the vertex type need not be finite.

No conservation, reversibility, geometric embedding,
spatial dimension, physical clock, or propagation speed
is assumed.

## Structural Reachability and Connected Components

Interaction reachability is defined as the reflexive
transitive closure of permitted interaction edges.

It is proved to be an equivalence relation.

The resulting equivalence classes form connected
components: any two components are either identical
or disjoint.

No permitted local interaction edge or finite permitted
path connects distinct components.

Furthermore, applying a common finite admissible event
history preserves agreement between configurations on
each connected component.

Consequently, initial differences confined to one
component cannot affect a distinct component under
the same admissible event history.

## Finite Walks and Derived Hop Distance

A length-indexed inductive relation describes finite
walks consisting of exactly a specified number of
permitted interaction edges.

Structural reachability is proved equivalent to the
existence of a finite interaction walk.

Every reachable pair admits a minimum walk length,
which is unique even when several shortest routes
may exist.

Interaction hop distance is defined as this minimum
length for reachable vertices and infinity for
unreachable vertices.

The resulting distance takes values in `WithTop Nat`.

It satisfies:

* Zero distance if and only if the vertices coincide.
* Symmetry.
* The triangle inequality.
* Finite distance exactly for reachable pairs.
* Infinite distance exactly for unreachable pairs.

Thus interaction hop distance is an extended metric
derived from relational connectivity rather than
from an assumed physical geometry.

## Propagation of Configuration Differences

The difference support of two configurations is the
set of vertices at which their oscillator states differ.

For a common admissible local event, a newly appearing
difference requires a pre-existing difference at one
of the event's participating vertices.

The new difference must itself occur at a participating
vertex and is therefore at most one interaction hop
from an existing difference.

This one-event bound extends by induction to arbitrary
finite admissible event histories.

After a common history containing `m` events, every
remaining configuration difference lies within at most
`m` interaction hops of some initial difference.

For configurations differing initially only at a
single possible source vertex, the final difference
support is contained in the interaction hop ball
centered at that source with radius `m`.

This establishes a finite-event propagation bound,
which may be represented as a structural interaction
causal cone.

## Principal Results

### Structural Reachability and Components

* `interaction_reachable_equivalence`:
  Interaction reachability is an equivalence relation
  on the vertices of the fixed network.

* `interaction_components_eq_or_disjoint`:
  Any two connected components are either identical
  or disjoint.

* `interaction_no_path_between_distinct_components`:
  No finite permitted interaction path connects
  vertices belonging to distinct components.

* `execute_history_preserves_component_agreement`:
  A common finite admissible event history preserves
  agreement between configurations on a component.

* `execute_history_no_cross_component_influence`:
  Initial differences confined to one component
  cannot affect a distinct component through a
  common admissible event history.

### Finite Walks and Extended Hop Distance

* `interaction_reachable_iff_exists_walk`:
  Structural reachability is equivalent to the
  existence of a finite permitted interaction walk.

* `interaction_shortest_walk_exists`:
  Every reachable pair admits a walk of minimum
  length, without requiring a finite network.

* `interaction_shortest_walk_length_unique`:
  The minimum walk length is unique, even when
  several shortest walks may exist.

* `interaction_hop_distance_eq_zero_iff`:
  Hop distance is zero exactly when the two
  vertices are identical.

* `interaction_hop_distance_symm`:
  Interaction hop distance is symmetric.

* `interaction_hop_distance_triangle`:
  Interaction hop distance satisfies the
  triangle inequality.

* `interaction_hop_distance_eq_top_iff`:
  Infinite hop distance is equivalent to
  structural unreachability.

* `interaction_hop_distance_ne_top_iff`:
  Finite hop distance is equivalent to
  structural reachability.

Together, these properties establish a derived
extended metric on the network vertices.

### One-Event Propagation

* `apply_admissible_event_no_new_difference`:
  A common local event cannot introduce new
  differences when its participating inputs agree.

* `apply_admissible_event_new_difference_requires_input_difference`:
  Every newly introduced difference requires a
  pre-existing difference at an event endpoint.

* `apply_admissible_event_new_difference_at_endpoint`:
  A newly introduced difference must occur at
  one of the event's participating vertices.

* `apply_admissible_event_new_difference_within_one_hop`:
  Every new difference lies within one interaction
  hop of a pre-existing difference.

* `apply_admissible_event_difference_within_one_hop`:
  Every difference remaining after a common event
  has an antecedent within at most one hop.

### Finite-History Propagation

* `execute_history_difference_bounded_by_length`:
  Every final difference after a common finite
  history has an initial antecedent within a
  hop distance bounded by the history length.

* `execute_history_single_source_no_influence_beyond_length`:
  A difference initially confined to one vertex
  cannot affect vertices farther away than the
  number of events in the common history.

* `execute_history_difference_support_subset_ball`:
  Final difference support from a single possible
  source is contained in the interaction hop ball
  whose radius equals the common history length.

## Development Structure

The principal constructions follow four related stages.

1. Structural reachability and component isolation

    InteractionReachable
      |
      +-- Reflexivity, symmetry, transitivity
      |
      +-- Reachability equivalence
      |
      +-- interaction_component
            |
            +-- Component partition
            |
            +-- Structural separation
            |
            +-- Operational component isolation
                  |
                  +-- Single-event agreement
                  |
                  +-- Finite-history agreement
                  |
                  +-- No cross-component influence

2. Finite walks and derived hop distance

    InteractionWalk
      |
      +-- Walk/reachability equivalence
      |
      +-- Walk extension and reversal
      |
      +-- Shortest-walk existence
      |
      +-- Minimum-length uniqueness
      |
      +-- interaction_hop_distance
            |
            +-- Identity of vertices
            |
            +-- Symmetry
            |
            +-- Walk concatenation
            |     |
            |     +-- Triangle inequality
            |
            +-- Finite/infinite distance
                  |
                  +-- Reachability characterization

3. One-event propagation of differences

    configuration_difference_support
      |
      +-- Equal inputs give equal outputs
      |
      +-- New differences require prior differences
      |
      +-- New differences occur at event endpoints
      |
      +-- One-hop bound for new differences
      |
      +-- One-hop bound for all final differences

4. Finite-history propagation bounds

    One-event propagation bound
      |
      +-- Induction over EventHistory
      |     |
      |     +-- Triangle inequality
      |     |
      |     +-- History-length propagation bound
      |
      +-- Single-source influence restriction
      |
      +-- interaction_hop_ball
            |
            +-- Final difference support contained
                within the history-length hop ball

The diagrams summarize the mathematical organization
of the development rather than the exact dependencies
of every individual Lean declaration.

The resulting propagation bound is expressed through
network hop distance and event count. Neither quantity
is identified with physical distance or elapsed time.

## Scope and Interpretation

Reachability expresses the existence of a permitted
interaction path, not actual transmission of influence.

Hop distance counts permitted edges, not physical
spatial separation.

History length counts executed interaction events,
not elapsed physical time.

The propagation bounds concern two configurations
subjected to the same fixed admissible event history.
They do not cover scheduling decisions that depend
on global configuration information.

The bounds identify where differences may propagate,
not where propagation must occur. Interactions may
preserve, eliminate, or spread existing differences.

The present results do not establish a relativistic
light cone, physical signal velocity, or emergent
spacetime geometry.

They provide a mathematical foundation for investigating
whether additional physical structure can be derived
from relational locality and oscillator interactions.
-/

namespace StandingWave

/-!
## Structural Reachability

We begin by examining connectivity in a fixed oscillator
network using only its permitted interaction edges.

Reachability is defined through finite paths of adjacent,
distinct vertices. Zero-edge paths establish reflexivity.

Symmetric adjacency allows permitted paths to be reversed,
while concatenation establishes transitivity.

These properties make reachability an equivalence relation.

The construction is purely structural. A permitted path
does not imply that interactions occur or that state
differences are actually transmitted.
-/

/--
Two vertices are reachable if they are identical, or
if a finite nonempty sequence of permitted interaction
edges connects them.

The permitted edge relation excludes self-loops,
but finite paths may revisit vertices or contain
cycles. Reachability does not require a simple path.

Reachability is a structural property of the fixed
network. It does not assert that an interaction occurs
or that causal influence is actually transmitted.
-/
def InteractionReachable
    {V : Type*} (N : OscillatorNetwork V)
    (u v : V) : Prop :=
  Relation.ReflTransGen
    (fun x y => N.adjacent x y ∧ x ≠ y) u v

/--
Every vertex is reachable from itself.

This follows from the reflexive closure of the
permitted interaction-edge relation and requires
no actual interaction or self-adjacency.

The corresponding path has zero edges.
-/
theorem interaction_reachable_refl
    {V : Type*} (N : OscillatorNetwork V) (u : V) :
    InteractionReachable N u u := by
  exact Relation.ReflTransGen.refl

/--
Two distinct adjacent vertices are reachable from
one another through a single permitted interaction edge.

This establishes one-step reachability without
asserting that an interaction actually occurs.
-/
theorem interaction_reachable_of_edge
    {V : Type*} (N : OscillatorNetwork V)
    {u v : V}
    (h : N.adjacent u v ∧ u ≠ v) :
    InteractionReachable N u v := by
  exact Relation.ReflTransGen.single h

/--
Interaction reachability is transitive.

If a finite permitted path connects `u` to `v`,
and another connects `v` to `w`, then their
concatenation connects `u` to `w`.

The paths may contain repeated vertices or cycles.
No interaction execution or temporal ordering
is assumed.
-/
theorem interaction_reachable_trans
    {V : Type*} (N : OscillatorNetwork V)
    {u v w : V}
    (huv : InteractionReachable N u v)
    (hvw : InteractionReachable N v w) :
    InteractionReachable N u w := by
  exact Relation.ReflTransGen.trans huv hvw

/--
Interaction reachability is symmetric.

If a finite permitted path connects `u` to `v`,
then reversing its edges gives a permitted path
from `v` to `u`.

This follows from symmetric network adjacency.
It does not assert that causal influence actually
propagates in either direction.
-/
theorem interaction_reachable_symm
    {V : Type*} (N : OscillatorNetwork V)
    {u v : V}
    (huv : InteractionReachable N u v) :
    InteractionReachable N v u := by
  induction huv with
  | refl =>
      exact Relation.ReflTransGen.refl
  | @tail b c hpath hbc ih =>
      have hcb : N.adjacent c b ∧ c ≠ b :=
        ⟨N.symmetric b c hbc.1, Ne.symm hbc.2⟩
      exact Relation.ReflTransGen.trans
        (Relation.ReflTransGen.single hcb) ih

/--
Interaction reachability is an equivalence relation
on the vertices of a fixed oscillator network.

Reflexivity follows from zero-edge reachability,
symmetry from reversible adjacency paths, and
transitivity from concatenation of finite paths.

The equivalence classes identify connected components
of the permitted interaction network.

No interactions, physical distances, or temporal
ordering are assumed.
-/
theorem interaction_reachable_equivalence
    {V : Type*} (N : OscillatorNetwork V) :
    Equivalence (InteractionReachable N) := by
  constructor
  · intro u
    exact interaction_reachable_refl N u
  · intro u v huv
    exact interaction_reachable_symm N huv
  · intro u v w huv hvw
    exact interaction_reachable_trans N huv hvw

/-!
## Connected Components

Interaction reachability defines an equivalence relation
on the vertices of a fixed oscillator network.

The equivalence classes form connected components.
Each component contains precisely those vertices
reachable from a chosen representative vertex.

Reachable vertices determine identical components.
Components sharing a vertex are equal, and distinct
components are disjoint.

Every permitted interaction edge connects vertices
belonging to the same component. Consequently,
no permitted interaction edge or finite interaction
path connects distinct components.

These results establish a partition of the network
based entirely on structural connectivity.

They do not require interaction execution, oscillator
state evolution, physical distance, or temporal
ordering.
-/

/--
The connected component containing a vertex in a
fixed oscillator network.

The component consists of all vertices reachable
from the given vertex through finite paths of
permitted interaction edges.

An isolated vertex forms a singleton component.

This is a structural definition. No interaction
execution, physical distance, or temporal ordering
is assumed.
-/
def interaction_component
    {V : Type*} (N : OscillatorNetwork V)
    (u : V) : Set V :=
  {v | InteractionReachable N u v}

/--
Reachable vertices determine the same connected component.

If `v` is reachable from `u`, then the two vertices
determine the same connected component.

The proof uses symmetry and transitivity of
interaction reachability.

No interaction execution or temporal ordering
is assumed.
-/
theorem interaction_component_eq_of_reachable
    {V : Type*} (N : OscillatorNetwork V)
    {u v : V}
    (huv : InteractionReachable N u v) :
    interaction_component N u = interaction_component N v := by
  ext w
  constructor
  · intro huw
    exact interaction_reachable_trans N
      (interaction_reachable_symm N huv) huw
  · intro hvw
    exact interaction_reachable_trans N huv hvw

/--
Two connected components sharing a vertex are equal.

If a vertex `w` is reachable from both `u` and `v`,
then `u` and `v` are mutually reachable and therefore
determine the same connected component.

Consequently, distinct connected components cannot
share a vertex.

This is a structural property of the fixed network.
No interaction execution or temporal ordering
is assumed.
-/
theorem interaction_components_eq_of_common_vertex
    {V : Type*} (N : OscillatorNetwork V)
    {u v w : V}
    (huw : w ∈ interaction_component N u)
    (hvw : w ∈ interaction_component N v) :
    interaction_component N u = interaction_component N v := by
  have huv : InteractionReachable N u v :=
    interaction_reachable_trans N
      huw (interaction_reachable_symm N hvw)
  exact interaction_component_eq_of_reachable N huv

/--
Distinct interaction components are disjoint.

If two connected components are not equal, they
cannot share any vertex. Otherwise, their common
vertex would establish equality of the components.

This property concerns structural connectivity in
a fixed network. It assumes no interaction execution,
physical distance, or temporal ordering.
-/
theorem interaction_components_disjoint_of_ne
    {V : Type*} (N : OscillatorNetwork V)
    {u v : V}
    (hne : interaction_component N u ≠
      interaction_component N v) :
    Disjoint (interaction_component N u)
      (interaction_component N v) := by
  apply Set.disjoint_left.mpr
  intro w huw hvw
  exact hne
    (interaction_components_eq_of_common_vertex
      N huw hvw)

/--
Any two interaction components are either equal
or disjoint.

If the components are equal, they represent the
same collection of mutually reachable vertices.

Otherwise, the previously established disjointness
theorem guarantees that they share no vertices.

This establishes the partition property of
connected components in a fixed oscillator network.
No interaction execution or temporal ordering
is assumed.
-/
theorem interaction_components_eq_or_disjoint
    {V : Type*} (N : OscillatorNetwork V)
    (u v : V) :
    interaction_component N u =
        interaction_component N v ∨
      Disjoint (interaction_component N u)
        (interaction_component N v) := by
  by_cases h :
      interaction_component N u =
        interaction_component N v
  · exact Or.inl h
  · exact Or.inr
      (interaction_components_disjoint_of_ne N h)

/--
The endpoints of a permitted interaction edge belong
to the same connected component.

A single permitted edge establishes reachability
between its endpoints. Reachable vertices determine
the same connected component.

Consequently, a permitted local interaction edge
cannot connect two distinct components of the
fixed network.

No actual interaction execution or temporal
ordering is assumed.
-/
theorem interaction_component_eq_of_edge
    {V : Type*} (N : OscillatorNetwork V)
    {u v : V}
    (h : N.adjacent u v ∧ u ≠ v) :
    interaction_component N u =
      interaction_component N v := by
  exact interaction_component_eq_of_reachable N
    (interaction_reachable_of_edge N h)

/--
Vertices belonging to distinct connected components
cannot be endpoints of the same permitted interaction edge.

If such an edge existed, its endpoints would determine
the same connected component, contradicting the
assumption that their components are distinct.

This establishes structural separation between
components of the fixed network. It does not require
interaction execution, physical distance, or time.
-/
theorem interaction_no_edge_between_distinct_components
    {V : Type*} (N : OscillatorNetwork V)
    {u v : V}
    (hne : interaction_component N u ≠
      interaction_component N v) :
    ¬ (N.adjacent u v ∧ u ≠ v) := by
  intro hedge
  exact hne (interaction_component_eq_of_edge N hedge)

/--
Vertices belonging to distinct connected components
cannot be connected by any finite path of permitted
interaction edges.

If such a path existed, its endpoints would be
reachable and therefore determine the same
connected component, contradicting their distinction.

This establishes structural path isolation in a
fixed oscillator network, without assuming
interaction execution or temporal ordering.
-/
theorem interaction_no_path_between_distinct_components
    {V : Type*} (N : OscillatorNetwork V)
    {u v : V}
    (hne : interaction_component N u ≠
      interaction_component N v) :
    ¬ InteractionReachable N u v := by
  intro huv
  exact hne (interaction_component_eq_of_reachable N huv)

/-!
## Operational Isolation of Components

The structural separation of connected components
constrains the execution of admissible local
interactions.

Every admissible event acts on two vertices belonging
to the same connected component. Consequently,
the event cannot modify vertices outside that
component.

More generally, applying the same admissible event
to two configurations preserves their agreement
throughout any connected component.

Induction over finite event histories extends this
property to arbitrary sequences of admissible
local interactions.

If two configurations initially agree throughout
a connected component, executing the same finite
event history preserves that agreement.

In particular, initial state differences confined
to one component cannot influence a distinct
component through a common admissible history.

These results establish operational noninterference
between disconnected components under fixed
local interaction histories.

They do not address globally state-dependent
event scheduling, which could introduce additional
dependencies outside the local update mechanism.

No conservation, reversibility, physical distance,
or temporal interpretation is assumed.
-/

/--
An admissible local interaction leaves every vertex
outside the connected component of its participants
unchanged.

Both participants belong to the same component
because they are joined by a permitted interaction
edge.

Consequently, a local event cannot directly modify
the oscillator state of a different component.

This result requires no conservation assumption,
physical distance, or temporal interpretation.
-/
theorem apply_admissible_event_outside_component
    {V : Type*} [DecidableEq V]
    (N : OscillatorNetwork V)
    (e : LocalInteractionEvent V)
    (h : AdmissibleLocalEvent N e)
    (X : OscillatorConfiguration V)
    (w : V)
    (hw : w ∉ interaction_component N e.u) :
    apply_admissible_event N e h X w = X w := by
  have hwu : w ≠ e.u := by
    intro heq
    apply hw
    rw [heq]
    exact interaction_reachable_refl N e.u
  have hwv : w ≠ e.v := by
    intro heq
    apply hw
    rw [heq]
    exact interaction_reachable_of_edge N h
  exact apply_admissible_event_other
    N e h X w hwu hwv

/--
Applying the same admissible event to two configurations
preserves their agreement on a connected component.

If the event acts within the component, both input
states agree and therefore produce identical outputs.

If the event acts outside the component, every vertex
inside the component remains unchanged.

This holds for arbitrary oscillator interactions and
does not require conservation, physical distance,
or a temporal interpretation.
-/
theorem apply_admissible_event_preserves_component_agreement
    {V : Type*} [DecidableEq V]
    (N : OscillatorNetwork V)
    (u : V)
    (e : LocalInteractionEvent V)
    (he : AdmissibleLocalEvent N e)
    (X Y : OscillatorConfiguration V)
    (hXY : ∀ w, w ∈ interaction_component N u →
      X w = Y w) :
    ∀ w, w ∈ interaction_component N u →
      apply_admissible_event N e he X w =
        apply_admissible_event N e he Y w := by
  intro w hw
  by_cases hwu : w = e.u
  · subst w
    have hev : e.v ∈ interaction_component N u :=
      interaction_reachable_trans N hw
        (interaction_reachable_of_edge N he)
    have hXu := hXY e.u hw
    have hXv := hXY e.v hev
    simp [apply_admissible_event, apply_pair_update, hXu, hXv]
  · by_cases hwv : w = e.v
    · subst w
      have heu : e.u ∈ interaction_component N u :=
        interaction_reachable_trans N hw
          (interaction_reachable_symm N
            (interaction_reachable_of_edge N he))
      have hXu := hXY e.u heu
      have hXv := hXY e.v hw
      simp [apply_admissible_event, apply_pair_update, hXu, hXv]
    · rw [apply_admissible_event_other N e he X w hwu hwv,
          apply_admissible_event_other N e he Y w hwu hwv]
      exact hXY w hw

/--
Executing the same finite admissible event history
preserves agreement between two configurations on
any connected component.

If two configurations initially agree at every
vertex of a component, each admissible event
preserves that agreement. Induction over the event
history therefore establishes agreement after
the complete history has executed.

The history may contain interactions both inside
and outside the component.

This result requires no conservation assumption,
physical distance, propagation speed, or clock.
-/
theorem execute_history_preserves_component_agreement
    {V : Type*} [DecidableEq V]
    (N : OscillatorNetwork V)
    (u : V)
    (H : EventHistory N)
    (X Y : OscillatorConfiguration V)
    (hXY : ∀ w, w ∈ interaction_component N u →
      X w = Y w) :
    ∀ w, w ∈ interaction_component N u →
      execute_history N H X w =
        execute_history N H Y w := by
  induction H generalizing X Y with
  | nil =>
      intro w hw
      simpa [execute_history_nil] using hXY w hw
  | cons event rest ih =>
      intro w hw
      rw [execute_history_cons, execute_history_cons]
      have hNext :
          ∀ z, z ∈ interaction_component N u →
            apply_admissible_event N event.1 event.2 X z =
              apply_admissible_event N event.1 event.2 Y z :=
        apply_admissible_event_preserves_component_agreement
          N u event.1 event.2 X Y hXY
      exact ih _ _ hNext w hw

/--
Initial state differences confined to one connected
component cannot affect a distinct component through
a common finite history of admissible local events.

The configurations may differ arbitrarily inside
the source component, but agree everywhere outside it.

Distinct components are disjoint, so the configurations
initially agree throughout the target component.
The history agreement theorem preserves this agreement.

The result assumes the same event history for both
configurations. No state-dependent scheduling rule
or physical temporal interpretation is assumed.
-/
theorem execute_history_no_cross_component_influence
    {V : Type*} [DecidableEq V]
    (N : OscillatorNetwork V)
    (u v : V)
    (H : EventHistory N)
    (X Y : OscillatorConfiguration V)
    (hne : interaction_component N u ≠
      interaction_component N v)
    (hXY : ∀ w, w ∉ interaction_component N u →
      X w = Y w) :
    ∀ w, w ∈ interaction_component N v →
      execute_history N H X w =
        execute_history N H Y w := by
  have hAgree :
      ∀ w, w ∈ interaction_component N v →
        X w = Y w := by
    intro w hwv
    apply hXY w
    intro hwu
    exact hne
      (interaction_components_eq_of_common_vertex
        N hwu hwv)
  exact execute_history_preserves_component_agreement
    N v H X Y hAgree

/-!
## Finite Interaction Walks

We refine structural reachability by introducing
finite interaction walks with explicitly specified
numbers of permitted interaction edges.

An inductive relation defines walks of exactly
`n` edges. Zero-edge walks establish reflexivity,
while longer walks are constructed by extending
existing walks with permitted edges.

Walks may revisit vertices and contain cycles.
They are not required to be simple paths.

We establish that every finite interaction walk
implies structural reachability, and conversely,
that every reachable pair admits a finite walk
of some length.

We also establish that walks may be extended,
reversed, and concatenated, with their lengths
behaving accordingly.

These results provide the foundation for deriving
shortest walk lengths and interaction hop distance.

Walk length counts permitted edges in the fixed
network. It does not represent physical distance,
elapsed time, or propagation speed.
-/
inductive InteractionWalk
    {V : Type*} (N : OscillatorNetwork V) :
    V → V → Nat → Prop where
  | zero (u : V) :
      InteractionWalk N u u 0
  | step {u v w : V} {n : Nat}
      (h : N.adjacent u v ∧ u ≠ v)
      (rest : InteractionWalk N v w n) :
      InteractionWalk N u w (n + 1)

/--
Every finite interaction walk establishes structural
reachability between its endpoints.

The proof proceeds by induction on the construction
of the walk.

A zero-edge walk establishes reflexive reachability.
A longer walk consists of one permitted edge followed
by a shorter walk, and establishes reachability
through transitivity.

The result introduces no physical distance,
propagation speed, or temporal ordering.
-/
theorem interaction_walk_implies_reachable
    {V : Type*} (N : OscillatorNetwork V)
    {u v : V} {n : Nat}
    (h : InteractionWalk N u v n) :
    InteractionReachable N u v := by
  induction h with
  | zero u =>
      exact interaction_reachable_refl N u
  | step hedge rest ih =>
      exact interaction_reachable_trans N
        (interaction_reachable_of_edge N hedge) ih

/--
Appending a permitted interaction edge to a finite
walk produces a walk whose length increases by one.

The proof proceeds by induction on the existing walk.

For a zero-edge walk, the appended edge forms a
one-edge walk.

For a longer walk, the first edge is retained
and the induction hypothesis extends the remaining
walk.

No physical distance or temporal interpretation
is assumed.
-/
theorem interaction_walk_append_edge
    {V : Type*} (N : OscillatorNetwork V)
    {u v : V} {n : Nat}
    (walk : InteractionWalk N u v n) :
    ∀ {w : V},
      (N.adjacent v w ∧ v ≠ w) →
      InteractionWalk N u w (n + 1) := by
  induction walk with
  | zero u =>
      intro w huw
      exact InteractionWalk.step huw
        (InteractionWalk.zero (N := N) w)
  | step hedge rest ih =>
      intro w hvw
      exact InteractionWalk.step hedge (ih hvw)

/--
Every structurally reachable pair of vertices admits
a finite interaction walk of some length.

The proof proceeds by induction on reachability.

Reflexive reachability gives a zero-edge walk.

When reachability is extended by a permitted edge,
the induction hypothesis supplies a finite walk to
the preceding vertex. The edge-extension theorem
then produces a walk to the new endpoint.

No physical distance, propagation speed, or
temporal interpretation is assumed.
-/
theorem interaction_reachable_implies_walk
    {V : Type*} (N : OscillatorNetwork V)
    {u v : V}
    (h : InteractionReachable N u v) :
    ∃ n : Nat, InteractionWalk N u v n := by
  induction h with
  | refl =>
      exact ⟨0, InteractionWalk.zero (N := N) _⟩
  | @tail b c hpath hbc ih =>
      obtain ⟨n, walk⟩ := ih
      exact ⟨n + 1,
        interaction_walk_append_edge N walk hbc⟩

/--
Structural reachability is equivalent to the existence
of a finite interaction walk of some length.

Every finite walk establishes reachability, and every
reachable pair admits a finite walk.

Thus the length-indexed walk relation characterizes
exactly the same connectivity as the original
reachability relation.

The walk length counts permitted interaction edges,
not physical distance or elapsed time.
-/
theorem interaction_reachable_iff_exists_walk
    {V : Type*} (N : OscillatorNetwork V)
    (u v : V) :
    InteractionReachable N u v ↔
      ∃ n : Nat, InteractionWalk N u v n := by
  constructor
  · intro h
    exact interaction_reachable_implies_walk N h
  · intro h
    obtain ⟨n, walk⟩ := h
    exact interaction_walk_implies_reachable N walk

/-!
## Shortest Walks and Extended Hop Distance

Having established the existence of finite walks
between reachable vertices, we investigate their
minimum possible lengths.

The well-ordering of the natural numbers guarantees
that every reachable pair admits a shortest walk.

Although multiple shortest walks may exist, their
minimum length is uniquely determined.

We define interaction hop distance as this minimum
length for reachable vertices and infinity for
unreachable vertices.

The resulting function takes values in `WithTop Nat`
and is well-defined even for infinite or disconnected
networks.

We establish the fundamental properties of this
derived distance:

* Zero distance occurs exactly between identical
  vertices.
* Distance is symmetric.
* Distance satisfies the triangle inequality.
* Finite distance is equivalent to reachability.
* Infinite distance is equivalent to unreachability.

Reversal and concatenation of finite walks provide
the essential constructions for proving symmetry
and the triangle inequality.

Together, these results establish an extended metric
derived entirely from permitted interaction edges.

The distance measures minimum graph-hop separation.
It is not assumed to represent physical distance,
elapsed time, or a propagation speed.
-/

/--
Every reachable pair of vertices admits a shortest
finite interaction walk.

Reachability guarantees that at least one finite
walk exists. The well-ordering of the natural
numbers then guarantees a least walk length.

The resulting walk length is no greater than the
length of any other permitted walk between the
same vertices.

No finiteness assumption on the network is required.
No physical distance or temporal interpretation
is imposed.
-/
theorem interaction_shortest_walk_exists
    {V : Type*} (N : OscillatorNetwork V)
    {u v : V}
    (h : InteractionReachable N u v) :
    ∃ n : Nat,
      InteractionWalk N u v n ∧
      ∀ m : Nat,
        InteractionWalk N u v m → n ≤ m := by
  classical
  have hex : ∃ n : Nat, InteractionWalk N u v n :=
    interaction_reachable_implies_walk N h
  let n : Nat := Nat.find hex
  refine ⟨n, ?_, ?_⟩
  · exact Nat.find_spec hex
  · intro m hm
    exact Nat.find_min' hex hm

/--
The length of a shortest interaction walk is unique.

Two walks may follow different routes while both
having minimum length.

If two natural numbers each satisfy the shortest-walk
condition between the same endpoints, they must be
equal by antisymmetry of the natural-number ordering.

This establishes uniqueness of minimum length,
not uniqueness of the walk itself.
-/
theorem interaction_shortest_walk_length_unique
    {V : Type*} (N : OscillatorNetwork V)
    {u v : V} {n₁ n₂ : Nat}
    (h₁ : InteractionWalk N u v n₁ ∧
      ∀ m : Nat,
        InteractionWalk N u v m → n₁ ≤ m)
    (h₂ : InteractionWalk N u v n₂ ∧
      ∀ m : Nat,
        InteractionWalk N u v m → n₂ ≤ m) :
    n₁ = n₂ := by
  apply Nat.le_antisymm
  · exact h₁.2 n₂ h₂.1
  · exact h₂.2 n₁ h₁.1

/--
The minimum number of permitted interaction edges
connecting two vertices of a fixed oscillator network.

For reachable vertices, the distance is the length
of a shortest finite interaction walk.

For unreachable vertices, the distance is infinity.

The minimum exists by well-ordering of the natural
numbers, and its value is unique.

The definition is noncomputable because no finite
search algorithm or decidable reachability procedure
has been assumed for arbitrary networks.

Hop distance is a structural quantity. It does not
represent physical distance, elapsed time, or a
propagation speed.
-/
noncomputable def interaction_hop_distance
    {V : Type*} (N : OscillatorNetwork V)
    (u v : V) : WithTop Nat := by
  classical
  exact if h : InteractionReachable N u v then
    (Nat.find
      (interaction_reachable_implies_walk N h) :
        WithTop Nat)
  else
    ⊤

/--
The interaction hop distance from any vertex to
itself is zero.

A zero-edge walk always exists from a vertex to
itself. Since walk lengths are natural numbers,
no shorter walk is possible.

Thus the minimum walk length is zero.

This is a structural property of the network and
does not refer to physical distance or time.
-/
theorem interaction_hop_distance_self
    {V : Type*} (N : OscillatorNetwork V)
    (u : V) :
    interaction_hop_distance N u u = 0 := by
  classical
  have h : InteractionReachable N u u :=
    interaction_reachable_refl N u
  have hex : ∃ n : Nat, InteractionWalk N u u n :=
    interaction_reachable_implies_walk N h
  have hmin : Nat.find hex = 0 := by
    apply Nat.eq_zero_of_le_zero
    exact Nat.find_min' hex
      (InteractionWalk.zero (N := N) u)
  simp only [interaction_hop_distance, dite_eq_left h]
  exact congrArg (fun n : Nat => (n : WithTop Nat)) hmin

/--
Zero interaction hop distance implies that the
two vertices are identical.

A finite hop distance of zero means the shortest
permitted walk has length zero.

By the inductive definition of InteractionWalk, a
zero-edge walk can only connect a vertex to itself.

Thus distinct vertices cannot have zero hop distance,
regardless of their adjacency or connectivity.

No physical distance or temporal interpretation
is assumed.
-/
theorem interaction_hop_distance_zero_implies_eq
    {V : Type*} (N : OscillatorNetwork V)
    {u v : V}
    (hzero : interaction_hop_distance N u v = 0) :
    u = v := by
  classical
  by_cases h : InteractionReachable N u v
  · have hmin :
        Nat.find (interaction_reachable_implies_walk N h) = 0 := by
      have hzero' :
          ((Nat.find
            (interaction_reachable_implies_walk N h) : Nat) :
              WithTop Nat) = 0 := by
        simpa only [interaction_hop_distance, dite_eq_left h]
          using hzero
      exact_mod_cast hzero'
    have hwalk : InteractionWalk N u v 0 := by
      simpa only [hmin] using
        (Nat.find_spec
          (interaction_reachable_implies_walk N h))
    cases hwalk with
    | zero _ => rfl
  · simp [interaction_hop_distance, h] at hzero

/--
Interaction hop distance is zero if and only if
its two endpoints are the same vertex.

Reflexivity supplies a zero-edge walk from every
vertex to itself.

Conversely, a zero minimum walk length requires
a zero-edge walk, whose endpoints must coincide.

This establishes the identity-of-indiscernibles
property for interaction hop distance.
-/
theorem interaction_hop_distance_eq_zero_iff
    {V : Type*} (N : OscillatorNetwork V)
    (u v : V) :
    interaction_hop_distance N u v = 0 ↔ u = v := by
  constructor
  · intro h
    exact interaction_hop_distance_zero_implies_eq N h
  · intro h
    subst v
    exact interaction_hop_distance_self N u

/--
Every finite interaction walk can be reversed
without changing its length.

The proof proceeds by induction on the walk.

A zero-edge walk reverses to itself.

For a longer walk, the induction hypothesis
reverses the remaining walk. Symmetry of network
adjacency then permits the original first edge
to be appended in the opposite direction.

The reversal preserves the number of edges.

This is a structural result and assumes no
physical distance or temporal interpretation.
-/
theorem interaction_walk_reverse
    {V : Type*} (N : OscillatorNetwork V)
    {u v : V} {n : Nat}
    (walk : InteractionWalk N u v n) :
    InteractionWalk N v u n := by
  induction walk with
  | zero u =>
      exact InteractionWalk.zero (N := N) u
  | @step a b c k hedge rest ih =>
      have hba : N.adjacent b a ∧ b ≠ a :=
        ⟨N.symmetric a b hedge.1, Ne.symm hedge.2⟩
      exact interaction_walk_append_edge N ih hba

/--
Interaction hop distance is symmetric.

For reachable vertices, reversing a shortest
interaction walk produces a walk of the same length
in the opposite direction.

Minimality in both directions establishes equality
of the shortest walk lengths.

For unreachable vertices, symmetry of reachability
ensures that both distances are infinite.

This property follows from symmetric network
adjacency, without assuming physical geometry,
propagation speed, or time.
-/
theorem interaction_hop_distance_symm
    {V : Type*} (N : OscillatorNetwork V)
    (u v : V) :
    interaction_hop_distance N u v =
      interaction_hop_distance N v u := by
  classical
  by_cases h : InteractionReachable N u v
  · have h' : InteractionReachable N v u :=
      interaction_reachable_symm N h
    have hex_uv :
        ∃ n : Nat, InteractionWalk N u v n :=
      interaction_reachable_implies_walk N h
    have hex_vu :
        ∃ n : Nat, InteractionWalk N v u n :=
      interaction_reachable_implies_walk N h'
    have hmin :
        Nat.find hex_uv = Nat.find hex_vu := by
      apply Nat.le_antisymm
      · exact Nat.find_min' hex_uv
          (interaction_walk_reverse N
            (Nat.find_spec hex_vu))
      · exact Nat.find_min' hex_vu
          (interaction_walk_reverse N
            (Nat.find_spec hex_uv))
    simp only [interaction_hop_distance,
      dite_eq_left h, dite_eq_left h']
    exact congrArg
      (fun n : Nat => (n : WithTop Nat)) hmin
  · have h' : ¬ InteractionReachable N v u := by
      intro hvu
      exact h (interaction_reachable_symm N hvu)
    simp [interaction_hop_distance, h, h']

/--
Two finite interaction walks with a common endpoint
can be concatenated.

A walk of `n` permitted edges from `u` to `v`,
followed by a walk of `m` permitted edges from
`v` to `w`, produces a walk of `n + m` edges
from `u` to `w`.

The proof uses induction on the first walk.

No network symmetry, physical distance, or
temporal interpretation is required.
-/
theorem interaction_walk_concat
    {V : Type*} (N : OscillatorNetwork V)
    {u v : V} {n : Nat}
    (first : InteractionWalk N u v n) :
    ∀ {w : V} {m : Nat},
      InteractionWalk N v w m →
        InteractionWalk N u w (n + m) := by
  induction first with
  | zero u =>
      intro w m second
      simpa using second
  | step hedge rest ih =>
      intro w m second
      have htail := ih second
      have hcombined := InteractionWalk.step hedge htail
      simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm]
        using hcombined

/--
The interaction hop distance between two vertices
is no greater than the length of any finite
interaction walk connecting them.

The existence of the walk establishes reachability,
so the hop distance is finite.

By definition, the hop distance is the minimum
length among all permitted connecting walks.

Consequently, it cannot exceed the length of the
given walk.

No physical distance or temporal interpretation
is assumed.
-/
theorem interaction_hop_distance_le_of_walk
    {V : Type*} (N : OscillatorNetwork V)
    {u v : V} {n : Nat}
    (walk : InteractionWalk N u v n) :
    interaction_hop_distance N u v ≤
      (n : WithTop Nat) := by
  classical
  have h : InteractionReachable N u v :=
    interaction_walk_implies_reachable N walk
  have hmin :
      Nat.find (interaction_reachable_implies_walk N h) ≤ n :=
    Nat.find_min'
      (interaction_reachable_implies_walk N h) walk
  simp only [interaction_hop_distance, dite_eq_left h]
  exact_mod_cast hmin

/--
Interaction hop distance satisfies the triangle inequality.

If both intermediate distances are finite, shortest
walks from `u` to `v` and from `v` to `w` can be
concatenated. The resulting walk has length equal
to the sum of their minimum lengths.

The shortest distance from `u` to `w` cannot
exceed the length of this concatenated walk.

If either intermediate distance is infinite,
the inequality follows from extended arithmetic.

Thus the triangle inequality holds even for
disconnected networks.

No physical distance, propagation speed, or
temporal interpretation is assumed.
-/
theorem interaction_hop_distance_triangle
    {V : Type*} (N : OscillatorNetwork V)
    (u v w : V) :
    interaction_hop_distance N u w ≤
      interaction_hop_distance N u v +
        interaction_hop_distance N v w := by
  classical
  by_cases huv : InteractionReachable N u v
  · by_cases hvw : InteractionReachable N v w
    · let p : Nat :=
        Nat.find (interaction_reachable_implies_walk N huv)
      let q : Nat :=
        Nat.find (interaction_reachable_implies_walk N hvw)
      have hp : InteractionWalk N u v p :=
        Nat.find_spec
          (interaction_reachable_implies_walk N huv)
      have hq : InteractionWalk N v w q :=
        Nat.find_spec
          (interaction_reachable_implies_walk N hvw)
      have hconcat : InteractionWalk N u w (p + q) :=
        interaction_walk_concat N hp hq
      have hbound :
          interaction_hop_distance N u w ≤
            ((p + q : Nat) : WithTop Nat) :=
        interaction_hop_distance_le_of_walk N hconcat
      have hduv :
          interaction_hop_distance N u v =
            (p : WithTop Nat) := by
        simp only [interaction_hop_distance,
          dite_eq_left huv]
        rfl
      have hdvw :
          interaction_hop_distance N v w =
            (q : WithTop Nat) := by
        simp only [interaction_hop_distance,
          dite_eq_left hvw]
        rfl
      rw [hduv, hdvw]
      calc
        interaction_hop_distance N u w ≤
            ((p + q : Nat) : WithTop Nat) :=
          hbound
        _ = (p : WithTop Nat) +
              (q : WithTop Nat) := by
          simp
    · have hinf :
          interaction_hop_distance N v w = ⊤ := by
        simp [interaction_hop_distance, hvw]
      rw [hinf]
      simp
  · have hinf :
        interaction_hop_distance N u v = ⊤ := by
      simp [interaction_hop_distance, huv]
    rw [hinf]
    simp

/--
Interaction hop distance is infinite if and only if
the two vertices are structurally unreachable.

Reachable vertices admit finite interaction walks,
so their shortest walk length is a natural number.

Unreachable vertices have infinite hop distance
by definition.

Thus infinite distance characterizes precisely
the separation of disconnected components.

No physical interpretation of infinity is assumed.
-/
theorem interaction_hop_distance_eq_top_iff
    {V : Type*} (N : OscillatorNetwork V)
    (u v : V) :
    interaction_hop_distance N u v = ⊤ ↔
      ¬ InteractionReachable N u v := by
  classical
  constructor
  · intro htop hreach
    simp [interaction_hop_distance, hreach] at htop
  · intro hnot
    simp [interaction_hop_distance, hnot]

/--
Interaction hop distance is finite if and only if
the two vertices are structurally reachable.

Reachable vertices admit a finite interaction walk,
and consequently have a finite minimum hop distance.

Unreachable vertices have infinite hop distance.

This characterizes connectivity through the
derived distance function without introducing
physical distance, propagation speed, or time.
-/
theorem interaction_hop_distance_ne_top_iff
    {V : Type*} (N : OscillatorNetwork V)
    (u v : V) :
    interaction_hop_distance N u v ≠ ⊤ ↔
      InteractionReachable N u v := by
  constructor
  · intro hfinite
    by_contra hnot
    exact hfinite
      ((interaction_hop_distance_eq_top_iff N u v).2 hnot)
  · intro hreach htop
    exact
      ((interaction_hop_distance_eq_top_iff N u v).1 htop)
        hreach

/-!
## Configuration Differences and One-Event Propagation

We investigate how differences between oscillator
configurations behave under admissible local interactions.

The difference support of two configurations consists
of those vertices at which their oscillator states
are unequal.

Applying the same admissible interaction to identical
input states produces identical outputs. Consequently,
an event cannot introduce new configuration differences
when its participating vertices initially agree.

More generally, any newly appearing difference must
have a pre-existing difference at an interaction
endpoint, and the new difference must itself occur
at one of the event's participating vertices.

Since the participants are connected by a permitted
interaction edge, every newly introduced difference
lies within one interaction hop of an existing
difference.

This bound also applies to differences that persist
through an event, using zero hop distance for an
unchanged source vertex.

These results establish a one-event propagation
constraint derived from relational locality.

They do not require conservation, reversibility,
physical distance, or a propagation speed.

The bound permits differences to propagate but
does not require propagation to occur. An interaction
may preserve, spread, or eliminate differences.
-/

/--
The set of vertices at which two oscillator
configurations have different states.

A vertex belongs to the difference support exactly
when the configurations assign unequal oscillator
states to that vertex.

The definition introduces no notion of physical
distance, propagation speed, or time.
-/
def configuration_difference_support
    {V : Type*}
    (X Y : OscillatorConfiguration V) : Set V :=
  {w | X w ≠ Y w}

/--
An admissible local event cannot introduce new
configuration differences when its participating
vertices initially agree.

If both configurations supply identical states
at the event's endpoints, the interaction produces
identical outputs at those endpoints.

All nonparticipating vertices retain their original
states.

Consequently, the difference support after the
common event is contained in the original
difference support.

No conservation, reversibility, physical distance,
or temporal interpretation is assumed.
-/
theorem apply_admissible_event_no_new_difference
    {V : Type*} [DecidableEq V]
    (N : OscillatorNetwork V)
    (e : LocalInteractionEvent V)
    (he : AdmissibleLocalEvent N e)
    (X Y : OscillatorConfiguration V)
    (hu : X e.u = Y e.u)
    (hv : X e.v = Y e.v) :
    configuration_difference_support
        (apply_admissible_event N e he X)
        (apply_admissible_event N e he Y)
      ⊆ configuration_difference_support X Y := by
  intro w hdiff
  change X w ≠ Y w
  intro hw
  apply hdiff
  simp [apply_admissible_event, apply_pair_update,
    hu, hv, hw]

/--
A newly introduced configuration difference requires
a pre-existing difference at an interaction endpoint.

Suppose the same admissible event is applied to
two configurations, producing a disagreement at
a vertex where the configurations previously agreed.

Then at least one participating vertex must have
had different input states in the two configurations.

The proof follows from the previously established
result that equal interaction inputs cannot
introduce new configuration differences.

This establishes a necessary condition for the
propagation of state differences through local events.

No conservation, reversibility, physical distance,
or temporal interpretation is assumed.
-/
theorem apply_admissible_event_new_difference_requires_input_difference
    {V : Type*} [DecidableEq V]
    (N : OscillatorNetwork V)
    (e : LocalInteractionEvent V)
    (he : AdmissibleLocalEvent N e)
    (X Y : OscillatorConfiguration V)
    (w : V)
    (hnew :
      w ∈ configuration_difference_support
        (apply_admissible_event N e he X)
        (apply_admissible_event N e he Y))
    (hold :
      w ∉ configuration_difference_support X Y) :
    e.u ∈ configuration_difference_support X Y ∨
      e.v ∈ configuration_difference_support X Y := by
  by_cases hu : X e.u = Y e.u
  · by_cases hv : X e.v = Y e.v
    · have hsubset :=
        apply_admissible_event_no_new_difference
          N e he X Y hu hv
      exact False.elim (hold (hsubset hnew))
    · exact Or.inr hv
  · exact Or.inl hu

/--
A newly introduced configuration difference must
occur at one of the participating vertices of
the admissible local event.

An admissible event leaves every nonparticipating
vertex unchanged.

Consequently, a vertex at which the configurations
previously agreed cannot develop a disagreement
unless it participates in the event.

This establishes that new differences have local
support, independently of the particular
oscillator interaction.

No conservation, reversibility, physical distance,
or temporal interpretation is assumed.
-/
theorem apply_admissible_event_new_difference_at_endpoint
    {V : Type*} [DecidableEq V]
    (N : OscillatorNetwork V)
    (e : LocalInteractionEvent V)
    (he : AdmissibleLocalEvent N e)
    (X Y : OscillatorConfiguration V)
    (w : V)
    (hnew :
      w ∈ configuration_difference_support
        (apply_admissible_event N e he X)
        (apply_admissible_event N e he Y))
    (hold :
      w ∉ configuration_difference_support X Y) :
    w = e.u ∨ w = e.v := by
  by_cases hwu : w = e.u
  · exact Or.inl hwu
  · by_cases hwv : w = e.v
    · exact Or.inr hwv
    · have hsame : X w = Y w := by
        by_contra hdiff
        exact hold hdiff
      have hnext :
          apply_admissible_event N e he X w =
            apply_admissible_event N e he Y w := by
        rw [
          apply_admissible_event_other N e he X w hwu hwv,
          apply_admissible_event_other N e he Y w hwu hwv
        ]
        exact hsame
      exact False.elim (hnew hnext)

/--
Every newly introduced configuration difference lies
within one interaction hop of a pre-existing difference.

If applying the same admissible local event to two
configurations creates a disagreement at a vertex
where they previously agreed, then at least one
participating vertex already had differing states.

The newly differing vertex must also be a participant.

Since the event's endpoints are connected by a
permitted edge, the new disagreement is at most
one hop from an existing disagreement.

This establishes a one-event propagation bound
without assuming conservation, physical distance,
propagation speed, or elapsed time.
-/
theorem apply_admissible_event_new_difference_within_one_hop
    {V : Type*} [DecidableEq V]
    (N : OscillatorNetwork V)
    (e : LocalInteractionEvent V)
    (he : AdmissibleLocalEvent N e)
    (X Y : OscillatorConfiguration V)
    (w : V)
    (hnew :
      w ∈ configuration_difference_support
        (apply_admissible_event N e he X)
        (apply_admissible_event N e he Y))
    (hold :
      w ∉ configuration_difference_support X Y) :
    ∃ z : V,
      z ∈ configuration_difference_support X Y ∧
      interaction_hop_distance N z w ≤
        (1 : WithTop Nat) := by
  have hsource :=
    apply_admissible_event_new_difference_requires_input_difference
      N e he X Y w hnew hold
  have htarget :=
    apply_admissible_event_new_difference_at_endpoint
      N e he X Y w hnew hold
  rcases hsource with hsource | hsource
  · refine ⟨e.u, hsource, ?_⟩
    rcases htarget with hw | hw
    · rw [hw, interaction_hop_distance_self]
      exact zero_le
    · rw [hw]
      have hwalk : InteractionWalk N e.u e.v 1 :=
        InteractionWalk.step he
          (InteractionWalk.zero (N := N) e.v)
      exact interaction_hop_distance_le_of_walk N hwalk
  · refine ⟨e.v, hsource, ?_⟩
    rcases htarget with hw | hw
    · rw [hw]
      have hrev : N.adjacent e.v e.u ∧ e.v ≠ e.u :=
        ⟨N.symmetric e.u e.v he.1, Ne.symm he.2⟩
      have hwalk : InteractionWalk N e.v e.u 1 :=
        InteractionWalk.step hrev
          (InteractionWalk.zero (N := N) e.u)
      exact interaction_hop_distance_le_of_walk N hwalk
    · rw [hw, interaction_hop_distance_self]
      exact zero_le

/--
Every configuration difference remaining after a
common admissible local event lies within one
interaction hop of a pre-existing difference.

If the difference already existed, the same vertex
provides a source at zero hop distance.

Otherwise, the previously established one-hop
propagation theorem supplies a pre-existing
difference within one permitted interaction edge.

This establishes a uniform one-event propagation
bound for both persistent and newly created
differences.

No physical distance, propagation speed, or
elapsed time is assumed.
-/
theorem apply_admissible_event_difference_within_one_hop
    {V : Type*} [DecidableEq V]
    (N : OscillatorNetwork V)
    (e : LocalInteractionEvent V)
    (he : AdmissibleLocalEvent N e)
    (X Y : OscillatorConfiguration V)
    (w : V)
    (hfinal :
      w ∈ configuration_difference_support
        (apply_admissible_event N e he X)
        (apply_admissible_event N e he Y)) :
    ∃ z : V,
      z ∈ configuration_difference_support X Y ∧
      interaction_hop_distance N z w ≤
        (1 : WithTop Nat) := by
  by_cases hold :
      w ∈ configuration_difference_support X Y
  · refine ⟨w, hold, ?_⟩
    rw [interaction_hop_distance_self]
    exact zero_le
  · exact
      apply_admissible_event_new_difference_within_one_hop
        N e he X Y w hfinal hold

/-!
## Finite-History Propagation Bounds

We extend the one-event propagation constraint to
arbitrary finite histories of admissible local interactions.

The central result establishes that every configuration
difference remaining after a common history of `m`
events lies within at most `m` interaction hops of
some initial configuration difference.

The proof proceeds by induction over the event history.
Each event contributes at most one additional hop,
and the triangle inequality combines successive
propagation bounds.

For configurations differing initially only at a
single possible source vertex, this establishes that
vertices farther than the history length from that
source cannot be affected by the initial difference.

We introduce interaction hop balls to represent the
sets of vertices within a specified number of
permitted interaction edges.

The final difference support is proved to be
contained within the hop ball centered at the
initial source, with radius equal to the number
of events in the common history.

This containment establishes a structural
interaction causal cone: a bound on the region
in which an initial difference may have an effect.

The cone represents possible influence rather
than necessary propagation. Individual interactions
may preserve, eliminate, or spread differences.

The results assume that both configurations undergo
the same fixed admissible event history. They do
not address globally state-dependent event scheduling.

History length counts interaction events rather
than elapsed physical time.

Consequently, the derived propagation bound does
not establish a physical velocity, relativistic
light cone, or spacetime geometry.
-/

/--
After a common finite history of admissible local
interactions, every remaining configuration difference
lies within at most as many interaction hops of an
initial difference as there are events in the history.

The proof proceeds by induction on the history.

For an empty history, any difference is already
present initially and has zero distance from itself.

For a nonempty history, the first event produces
an intermediate pair of configurations. The induction
hypothesis traces a final difference back to an
intermediate difference within the remaining number
of hops.

The one-event propagation theorem traces that
intermediate difference back to an initial difference
within one additional hop.

The triangle inequality combines these bounds.

This result assumes a common fixed event history.
The event count is not interpreted as elapsed
physical time or a propagation speed.
-/
theorem execute_history_difference_bounded_by_length
    {V : Type*} [DecidableEq V]
    (N : OscillatorNetwork V)
    (H : EventHistory N)
    (X Y : OscillatorConfiguration V) :
    ∀ w : V,
      w ∈ configuration_difference_support
        (execute_history N H X)
        (execute_history N H Y) →
      ∃ z : V,
        z ∈ configuration_difference_support X Y ∧
        interaction_hop_distance N z w ≤
          (H.length : WithTop Nat) := by
  induction H generalizing X Y with
  | nil =>
      intro w hw
      simp only [execute_history_nil] at hw
      refine ⟨w, hw, ?_⟩
      simp [interaction_hop_distance_self]
  | cons event rest ih =>
      intro w hw
      rw [execute_history_cons, execute_history_cons] at hw
      obtain ⟨z, hz, hzw⟩ := ih
        (apply_admissible_event N event.1 event.2 X)
        (apply_admissible_event N event.1 event.2 Y)
        w hw
      obtain ⟨a, ha, haz⟩ :=
        apply_admissible_event_difference_within_one_hop
          N event.1 event.2 X Y z hz
      refine ⟨a, ha, ?_⟩
      have hbound :
          interaction_hop_distance N a w ≤
            (1 : WithTop Nat) +
              (rest.length : WithTop Nat) := by
        calc
          interaction_hop_distance N a w ≤
              interaction_hop_distance N a z +
                interaction_hop_distance N z w :=
            interaction_hop_distance_triangle N a z w
          _ ≤ (1 : WithTop Nat) +
                (rest.length : WithTop Nat) :=
            add_le_add haz hzw
      simpa [List.length_cons, add_comm] using hbound

/--
An initial configuration difference confined to a
single vertex cannot influence a vertex farther away
than the number of events in a common finite history.

The configurations agree initially everywhere except
possibly at the source vertex `u`.

If the hop distance from `u` to `w` exceeds the
history length, the final configurations must agree
at `w`.

The proof follows from the general finite-propagation
bound: any final difference must have an initial
antecedent within the history-length distance bound.

This is an operational propagation constraint.
Event count is not physical elapsed time.
-/
theorem execute_history_single_source_no_influence_beyond_length
    {V : Type*} [DecidableEq V]
    (N : OscillatorNetwork V)
    (H : EventHistory N)
    (X Y : OscillatorConfiguration V)
    (u w : V)
    (honly : ∀ z : V, z ≠ u → X z = Y z)
    (hfar :
      (H.length : WithTop Nat) <
        interaction_hop_distance N u w) :
    execute_history N H X w =
      execute_history N H Y w := by
  by_contra hdiff
  have hfinal :
      w ∈ configuration_difference_support
        (execute_history N H X)
        (execute_history N H Y) := hdiff
  obtain ⟨z, hz, hzw⟩ :=
    execute_history_difference_bounded_by_length
      N H X Y w hfinal
  have hzu : z = u := by
    by_contra hne
    exact hz (honly z hne)
  subst z
  exact (not_le_of_gt hfar) hzw

/--
The interaction hop ball centered at vertex `u`
with natural-number radius `m`.

The ball contains precisely those vertices whose
minimum interaction hop distance from `u` is at
most `m`.

Unreachable vertices are excluded from every
finite-radius ball because their distance is
infinite.

The radius counts permitted interaction edges,
not physical distance or elapsed time.
-/
def interaction_hop_ball
    {V : Type*} (N : OscillatorNetwork V)
    (u : V) (m : Nat) : Set V :=
  {w | interaction_hop_distance N u w ≤
    (m : WithTop Nat)}

/--
The final difference support of two configurations
that initially differ only at one possible source
vertex is contained in the interaction hop ball
centered at that source, with radius equal to
the number of events in their common history.

Every final disagreement has an initial antecedent
within the history-length hop bound.

Since the initial configurations agree everywhere
except possibly at `u`, that antecedent must be `u`.

This establishes containment within a structural
interaction-propagation cone.

The event count is not physical elapsed time, and
the ball describes possible rather than necessary
influence.
-/
theorem execute_history_difference_support_subset_ball
    {V : Type*} [DecidableEq V]
    (N : OscillatorNetwork V)
    (H : EventHistory N)
    (X Y : OscillatorConfiguration V)
    (u : V)
    (honly : ∀ z : V, z ≠ u → X z = Y z) :
    configuration_difference_support
        (execute_history N H X)
        (execute_history N H Y)
      ⊆ interaction_hop_ball N u H.length := by
  intro w hfinal
  change interaction_hop_distance N u w ≤
    (H.length : WithTop Nat)
  obtain ⟨z, hz, hzw⟩ :=
    execute_history_difference_bounded_by_length
      N H X Y w hfinal
  have hzu : z = u := by
    by_contra hne
    exact hz (honly z hne)
  simpa only [hzu] using hzw


end StandingWave
