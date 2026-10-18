/-
Copyright (c) 2026 Wayne Brassem. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Wayne Brassem
-/

import StandingWave.Coupling

/-!
# Relational Locality and Event Dependencies

This module introduces locality into the abstract oscillator
framework without assuming a physical spacetime, spatial
coordinates, a metric, or a propagation speed.

Oscillator states are assigned to vertices of an abstract
network. A symmetric adjacency relation specifies which
pairs of vertices are permitted to interact.

The objective is to investigate the mathematical consequences
of restricted interactions, particularly the independence
of disjoint events and the dependency structures associated
with finite interaction histories.

Locality is introduced as a primitive relational property.
It is not derived from, or defined in terms of, geometric
proximity.

## Foundational Principle: Relational Locality

In this framework, locality refers exclusively to the
admissibility of direct interaction between vertices.

Two vertices are local to one another when the network
permits a direct interaction between them.

This notion is independent of physical distance.

In particular:

* Adjacency does not imply geometric proximity.

* Geometric separation does not prohibit adjacency.

* No metric or spatial embedding is assumed.

* No propagation time or limiting interaction speed
  is imposed by the adjacency relation.

* Adjacency permits interaction but does not require
  that an interaction occur or produce a state change.

Consequently, vertices that might eventually be interpreted
as widely separated in physical spacetime may nevertheless
be adjacent in the underlying interaction network.

Conversely, vertices that appear geometrically close in
a future representation need not be directly connected.

The distinction between relational and geometric locality
is deliberate and foundational.

Any connection between network adjacency and an emergent
geometry must be established through subsequent mathematical
development rather than imposed by definition.

## Mathematical Framework

### Networks and Configurations

An `OscillatorNetwork V` consists of a vertex type `V`
and a symmetric adjacency relation

    adjacent : V → V → Prop.

Symmetry is assumed:

    adjacent u v → adjacent v u.

Irreflexivity is not required of the network.

An oscillator configuration assigns one abstract oscillator
state to each vertex:

    OscillatorConfiguration V = V → OscillatorState.

No geometric positions are assigned to the vertices.

### Admissible Local Interactions

A local interaction event specifies two participating
vertices and an oscillator interaction acting on their
states.

An event is admissible when:

* Its two vertices are adjacent.
* Its two vertices are distinct.

The distinction between network adjacency and event
admissibility permits the network relation itself to
remain reflexive or non-reflexive.

Applying an admissible event may change the states of
its two participating vertices.

Every other vertex retains its previous state.

Thus local support is expressed through the restriction
of state changes to the participating vertices, without
reference to physical distance.

No conservation or reversibility assumption is imposed
on an arbitrary local interaction.

### Independent and Dependent Events

Two local events are structurally independent when their
participating vertex sets are disjoint.

Disjointness refers to shared participation, not to
geometric separation or to adjacency between the
participants of different events.

The module proves that disjoint local updates commute:

    apply e₁ (apply e₂ X) =
      apply e₂ (apply e₁ X).

This equality holds for arbitrary initial configurations
and arbitrary interactions supported on disjoint pairs.

Structural dependence is defined as the negation of
disjointness.

Thus two events are structurally dependent when they
share at least one participating vertex.

Structural dependence is symmetric, but it does not
impose an ordering between the events.

Furthermore, structural dependence does not imply that
the particular interactions fail to commute. It only
means that commutativity cannot be inferred from
disjointness alone.

### Finite Event Histories

An `EventHistory` is a finite list of admissible events.

Executing a history applies its events successively to
an initial configuration.

The list ordering specifies an operational evaluation
order. It does not introduce physical time, elapsed
duration, or a universal clock.

Histories are related by an elementary swap when two
consecutive disjoint events are exchanged, possibly
within an arbitrary common prefix and suffix.

`HistoryEquivalent` is the equivalence relation generated
by finitely many such swaps.

The module establishes:

    HistoryEquivalent H₁ H₂ →
      execute_history network H₁ X =
        execute_history network H₂ X.

Therefore, the resulting configuration is invariant
under permitted reorderings of independent events.

The converse is not assumed. Different histories may
produce identical configurations for reasons unrelated
to structural independence.

### Occurrence Dependencies and Precedence

Individual event occurrences are identified by their
positions within a finite history.

Occurrences must be distinguished from event values:
the same event may occur multiple times.

Two occurrences have a direct structural dependency
when the first has a smaller history index and the
corresponding events share a participating vertex.

The relation `HistoryPrecedes` is the transitive closure
of direct structural dependency.

The module proves that structural precedence is
transitive, irreflexive, and compatible with strict
ordering of occurrence indices.

Consequently, it defines a strict partial order on
the occurrences of each finite history.

This order describes structural dependencies imposed
by shared participation and the history's ordering.

It does not establish physical causation or information
transfer between occurrences.

In particular, two events may be disjoint yet related
by precedence through a chain of intermediate dependent
events.

### Correspondence Between Histories

A `HistoryOccurrenceEquiv` associates event occurrences
in two histories through a bijection of their indices.

The correspondence preserves the event label attached
to each occurrence.

A separate predicate, `PreservesHistoryPrecedence`,
expresses preservation of the dependency order in
both directions.

Identity correspondences, inverse correspondences,
and composition have been established, together with
their associated precedence-preservation properties.

The module also constructs an occurrence correspondence
for two histories differing by a single adjacent swap.

This construction uses reusable finite-index permutation
and list-lookup lemmas.

An arbitrary adjacent swap preserves event labels,
but preservation of structural precedence additionally
requires the exchanged events to be independent.

The module proves that equivalent finite event histories
possess isomorphic labeled structural dependency partial
orders. The occurrence correspondence preserves event
labels and structural precedence in both directions.

## Principal Results

* `apply_pair_update_other`:
  A pair update leaves every nonparticipating vertex
  unchanged.

* `apply_admissible_event_other`:
  An admissible event leaves all other vertices unchanged.

* `apply_pair_update_comm_of_disjoint`:
  Updates on disjoint vertex pairs commute.

* `apply_admissible_events_comm_of_disjoint`:
  Disjoint admissible events commute.

* `execute_history_append`:
  Concatenated histories execute by successive application.

* `execute_history_swap_disjoint`:
  Two consecutive disjoint events may be exchanged without
  changing the resulting configuration.

* `execute_history_swap_disjoint_context`:
  The same exchange is valid inside arbitrary finite
  prefixes and suffixes.

* `execute_history_eq_of_equivalent`:
  Equivalent histories produce identical final
  configurations for every initial configuration.

* `dependent_admissible_events_symm`:
  Structural dependence between events is symmetric.

* `history_direct_dependency_irrefl`:
  An occurrence cannot directly precede itself.

* `history_direct_dependency_asymm`:
  Direct structural dependency is asymmetric.

* `history_precedes_trans`:
  Structural precedence is transitive.

* `history_precedes_implies_lt`:
  Structural precedence implies strict index ordering.

* `history_precedes_irrefl`:
  Structural precedence contains no self-precedence.

* `adjacent_occurrence_swap_involutive`:
  Exchanging consecutive occurrence indices twice
  restores the original index.

* `getElem?_swap_after_prefix`:
  An adjacent list swap preserves optional lookup
  under the corresponding natural-number permutation.

* `adjacent_occurrence_swap_val`:
  The finite-index and natural-number swap operations
  agree on their index values.

* `get_swap_after_prefix`:
  The occurrence-index correspondence preserves
  list elements under an adjacent swap.

* `history_occurrence_equiv_swap`:
  An adjacent swap induces a label-preserving
  bijection between event occurrences.

* `history_occurrence_equiv_refl_preserves`:
  The identity correspondence preserves precedence.

* `history_occurrence_equiv_symm_preserves`:
  Inverting a precedence-preserving correspondence
  preserves precedence.

* `history_occurrence_equiv_trans_preserves`:
  Composing precedence-preserving correspondences
  preserves precedence.

* `history_equivalent_has_precedence_equiv`:
  Equivalent event histories possess isomorphic
  labeled structural dependency partial orders,
  preserving event labels and precedence.

## Development Structure

The principal constructions follow four related stages.

1. Relational networks and local updates

    OscillatorNetwork
      |
      +-- OscillatorConfiguration
      |
      +-- LocalInteractionEvent
            |
            +-- AdmissibleLocalEvent
            |
            +-- apply_admissible_event
                  |
                  +-- Local support
                  |
                  +-- Commutativity of disjoint events

2. Finite histories and equivalence

    AdmissibleEvent
      |
      +-- EventHistory
            |
            +-- execute_history
            |
            +-- HistorySwap
                  |
                  +-- HistoryEquivalent
                        |
                        +-- Execution invariance

3. Occurrence dependencies

    DependentAdmissibleEvents
      |
      +-- HistoryDirectDependency
            |
            +-- HistoryPrecedes
                  |
                  +-- Transitivity
                  |
                  +-- Irreflexivity
                  |
                  +-- Strict partial order

4. Correspondence of dependency structures

    HistoryOccurrenceEquiv
      |
      +-- Identity correspondence
      |
      +-- Inverse correspondence
      |
      +-- Composition of correspondences
      |
      +-- Adjacent occurrence swap
      |     |
      |     +-- Index permutation
      |     |
      |     +-- Event-label preservation
      |     |
      |     +-- Precedence preservation
      |
      +-- History equivalence
            |
            +-- Isomorphism of labeled
                dependency partial orders

These diagrams summarize the organization of the
development rather than the exact dependencies of
individual Lean declarations.

## Scope and Interpretation

The mathematical meaning of locality in this module
is restricted to admissible direct interaction.

The network is not assumed to represent a physical
spacetime or a geometric discretization of one.

No spatial dimension, distance function, metric,
light cone, limiting speed, or temporal coordinate
is introduced.

In particular, graph-theoretic locality must not be
identified automatically with relativistic locality
or the locality assumptions used in Bell's theorem.

Quantum entanglement provides one motivation for
distinguishing physical separation from underlying
relationships between systems.

However, neither entanglement nor quantum measurement
is modeled by the present oscillator configuration.

Network adjacency alone does not establish quantum
correlations or imply a physical signaling mechanism.

Possible interpretations involving higher-dimensional
relationships, non-geometric connectivity, or emergent
spacetime remain open questions rather than assumptions.

Similarly, event-history ordering is a mathematical
device for composing transformations.

The derived dependency order is a structural partial
order, not yet a physical causal order.

The present framework establishes that certain
reorderings of independent events preserve both the
event labels and, at the execution level, the resulting
configuration.

Equivalent histories have been proved to determine
isomorphic labeled structural dependency partial
orders. The converse characterization remains an open
question in the present development.

Whether any of these structures admit an interpretation
as physical causality, spacetime, or information
propagation remains a separate question requiring
further derivation and empirical justification.

Finally, this module considers finite sequential
histories. A general rule for simultaneous overlapping
interactions is not yet defined.

The underlying locality principle does not require
interactions to be fundamentally sequential; that
choice belongs to the present stage of development.
-/

namespace StandingWave

/--
An abstract network whose adjacency relation specifies
which pairs of vertices are permitted to interact.

Locality is relational rather than geometric: two
vertices are local to one another when direct interaction
between them is admissible, independently of any physical
distance or spatial embedding.

The adjacency relation is symmetric but is not required
to be irreflexive. Individual admissible interaction
events separately require distinct participating vertices.

Adjacency grants permission for interaction; it does not
imply that an interaction occurs or changes a state.

No metric, physical coordinate system, propagation speed,
or spacetime interpretation is assumed.
-/
structure OscillatorNetwork (V : Type*) where
  adjacent : V → V → Prop
  symmetric : ∀ u v, adjacent u v → adjacent v u

/--
A configuration assigns an oscillator state to every
site in an abstract network.

No spatial coordinates or temporal interpretation
are assumed.
-/
def OscillatorConfiguration (V : Type*) :=
  V → OscillatorState

/--
Apply a two-state interaction at two distinct sites.

The first output is assigned to `u`, the second to `v`,
and all other sites retain their previous states.

The distinctness condition prevents conflicting
assignments to the same site.

This definition does not itself require adjacency;
the network's adjacency constraint is imposed by
the subsequent definition of an admissible update.
-/
def apply_pair_update
    {V : Type*} [DecidableEq V]
    (X : OscillatorConfiguration V)
    (u v : V)
    (_huv : u ≠ v)
    (F : OscillatorInteraction) :
    OscillatorConfiguration V :=
  let output := F (X u, X v)
  fun w =>
    if w = u then output.1
    else if w = v then output.2
    else X w

/--
An elementary pair update leaves every nonparticipating
site unchanged.

This is the local-support property of the update
operation, independent of the interaction's effect
on its two participating states.
-/
theorem apply_pair_update_other
    {V : Type*} [DecidableEq V]
    (X : OscillatorConfiguration V)
    (u v w : V)
    (huv : u ≠ v)
    (F : OscillatorInteraction)
    (hwu : w ≠ u)
    (hwv : w ≠ v) :
    apply_pair_update X u v huv F w = X w := by
  simp [apply_pair_update, hwu, hwv]

/--
A proposed interaction between two sites of an oscillator network.

The event specifies an ordered pair of participating
vertices and an oscillator interaction.

The vertices are ordered because the first and second
outputs of an interaction may differ.

No adjacency or distinctness condition is imposed here.
Those conditions are expressed separately by admissibility.
-/
structure LocalInteractionEvent (V : Type*) where
  u : V
  v : V
  interaction : OscillatorInteraction

/--
An event is admissible when its two participating sites
are distinct and adjacent in the network.

Adjacency is supplied by the network structure and
is assumed symmetric.

Distinctness is required by our current two-site update
operation, but self-adjacency is not prohibited by the
network definition.

Admissibility imposes no conservation or non-increase
condition on the event's oscillator interaction.
-/
def AdmissibleLocalEvent
    {V : Type*}
    (network : OscillatorNetwork V)
    (event : LocalInteractionEvent V) : Prop :=
  network.adjacent event.u event.v ∧ event.u ≠ event.v

/--
Apply an admissible local interaction event to a network
configuration.

The event's interaction acts on the states at its two
participating vertices.

All other vertices retain their previous states.

The admissibility proof ensures that the participants
are distinct and adjacent. The adjacency property
determines whether the update is permitted; it does
not alter the oscillator interaction itself.
-/
def apply_admissible_event
    {V : Type*} [DecidableEq V]
    (network : OscillatorNetwork V)
    (event : LocalInteractionEvent V)
    (h : AdmissibleLocalEvent network event)
    (X : OscillatorConfiguration V) :
    OscillatorConfiguration V :=
  apply_pair_update X event.u event.v h.2 event.interaction


/--
Applying an admissible local event leaves every
nonparticipating vertex unchanged.

This follows from the previously established local-support
property of `apply_pair_update`.
-/
theorem apply_admissible_event_other
    {V : Type*} [DecidableEq V]
    (network : OscillatorNetwork V)
    (event : LocalInteractionEvent V)
    (h : AdmissibleLocalEvent network event)
    (X : OscillatorConfiguration V)
    (w : V)
    (hwu : w ≠ event.u)
    (hwv : w ≠ event.v) :
    apply_admissible_event network event h X w = X w := by
  exact apply_pair_update_other
    X event.u event.v w h.2 event.interaction hwu hwv

/--
Two local interaction events are disjoint when they
have no participating vertices in common.

This condition concerns the active participants of
the events, not the absence of adjacency between
their respective vertices.

Disjoint events operate on separate parts of a
configuration, even if edges connect their sites
elsewhere in the network.
-/
def DisjointLocalEvents
    {V : Type*}
    (e₁ e₂ : LocalInteractionEvent V) : Prop :=
  e₁.u ≠ e₂.u ∧
  e₁.u ≠ e₂.v ∧
  e₁.v ≠ e₂.u ∧
  e₁.v ≠ e₂.v

/--
Updates on two disjoint pairs of vertices commute.

Each update reads and modifies only its participating
vertices. Since the pairs are disjoint, neither update
changes the inputs used by the other.

The result holds for arbitrary oscillator interactions;
the interactions themselves need not commute.
-/
theorem apply_pair_update_comm_of_disjoint
    {V : Type*} [DecidableEq V]
    (X : OscillatorConfiguration V)
    (u v a b : V)
    (huv : u ≠ v)
    (hab : a ≠ b)
    (hua : u ≠ a)
    (hub : u ≠ b)
    (hva : v ≠ a)
    (hvb : v ≠ b)
    (F G : OscillatorInteraction) :
    apply_pair_update
        (apply_pair_update X u v huv F)
        a b hab G =
      apply_pair_update
        (apply_pair_update X a b hab G)
        u v huv F := by
  funext w
  simp only [apply_pair_update]
  split_ifs <;> subst_vars <;> simp_all

/--
Two disjoint admissible local events commute as
transformations of oscillator configurations.

The events may use arbitrary oscillator interactions,
including non-conservative and non-increasing ones.

Commutativity follows entirely from disjointness of
their participating vertices, not from any algebraic
commutativity of their interaction functions.

This establishes that exchanging the order of two
independent local events does not alter the resulting
configuration.
-/
theorem apply_admissible_events_comm_of_disjoint
    {V : Type*} [DecidableEq V]
    (network : OscillatorNetwork V)
    (e₁ e₂ : LocalInteractionEvent V)
    (h₁ : AdmissibleLocalEvent network e₁)
    (h₂ : AdmissibleLocalEvent network e₂)
    (hd : DisjointLocalEvents e₁ e₂)
    (X : OscillatorConfiguration V) :
    apply_admissible_event network e₂ h₂
        (apply_admissible_event network e₁ h₁ X) =
      apply_admissible_event network e₁ h₁
        (apply_admissible_event network e₂ h₂ X) := by
  unfold apply_admissible_event
  exact apply_pair_update_comm_of_disjoint
    X e₁.u e₁.v e₂.u e₂.v
    h₁.2 h₂.2
    hd.1 hd.2.1 hd.2.2.1 hd.2.2.2
    e₁.interaction e₂.interaction


/-!
## Finite Histories of Local Interactions

A finite history is a list of admissible local events.

The list specifies an order of application, but its
indices do not represent physical time or duration.

We investigate which changes in event ordering leave
the resulting configuration unchanged.
-/

/--
An admissible event in a given oscillator network.

The underlying event specifies the participating sites
and interaction. The accompanying proof establishes
adjacency and distinctness of the sites.
-/
def AdmissibleEvent
    {V : Type*}
    (network : OscillatorNetwork V) :=
  {e : LocalInteractionEvent V //
    AdmissibleLocalEvent network e}


/--
A finite ordered list of admissible local events.

The list order specifies the order in which interactions
are applied. No physical time interpretation is imposed.
-/
abbrev EventHistory
    {V : Type*}
    (network : OscillatorNetwork V) :=
  List (AdmissibleEvent network)

/--
Apply a finite history of admissible local events
to an oscillator configuration.

Each event operates on the configuration produced
by the preceding event.

The empty history leaves the configuration unchanged.

The ordering is operational rather than temporal:
histories with different list orders may produce
the same final configuration.
-/
def execute_history
    {V : Type*} [DecidableEq V]
    (network : OscillatorNetwork V)
    (history : EventHistory network)
    (X : OscillatorConfiguration V) :
    OscillatorConfiguration V :=
  history.foldl
    (fun current event =>
      apply_admissible_event
        network event.1 event.2 current)
    X

/--
Executing an empty event history leaves the initial
oscillator configuration unchanged.
-/
theorem execute_history_nil
    {V : Type*} [DecidableEq V]
    (network : OscillatorNetwork V)
    (X : OscillatorConfiguration V) :
    execute_history network [] X = X := by
  rfl

/--
Executing a history beginning with an event first
applies that event, then executes the remaining history.

This identity exposes the operational structure
of finite event histories.
-/
theorem execute_history_cons
    {V : Type*} [DecidableEq V]
    (network : OscillatorNetwork V)
    (event : AdmissibleEvent network)
    (history : EventHistory network)
    (X : OscillatorConfiguration V) :
    execute_history network (event :: history) X =
      execute_history network history
        (apply_admissible_event
          network event.1 event.2 X) := by
  rfl

/--
Executing concatenated histories is equivalent to
executing the first history and then the second.

This follows from the behavior of a left fold over
concatenated lists and introduces no temporal assumptions.
-/
theorem execute_history_append
    {V : Type*} [DecidableEq V]
    (network : OscillatorNetwork V)
    (H₁ H₂ : EventHistory network)
    (X : OscillatorConfiguration V) :
    execute_history network (H₁ ++ H₂) X =
      execute_history network H₂
        (execute_history network H₁ X) := by
  simp only [execute_history, List.foldl_append]

/--
Two admissible events are disjoint when their underlying
events have no participating vertices in common.
-/
def DisjointAdmissibleEvents
    {V : Type*}
    {network : OscillatorNetwork V}
    (e₁ e₂ : AdmissibleEvent network) : Prop :=
  DisjointLocalEvents e₁.1 e₂.1

/--
Swapping two consecutive disjoint admissible events
does not change the resulting configuration.

This follows from the commutativity of local updates
with disjoint participating vertices.

The underlying oscillator interactions need not commute.
-/
theorem execute_history_swap_disjoint
    {V : Type*} [DecidableEq V]
    (network : OscillatorNetwork V)
    (e₁ e₂ : AdmissibleEvent network)
    (hd : DisjointAdmissibleEvents e₁ e₂)
    (X : OscillatorConfiguration V) :
    execute_history network [e₁, e₂] X =
      execute_history network [e₂, e₁] X := by
  change
    apply_admissible_event network e₂.1 e₂.2
        (apply_admissible_event network e₁.1 e₁.2 X) =
      apply_admissible_event network e₁.1 e₁.2
        (apply_admissible_event network e₂.1 e₂.2 X)
  exact apply_admissible_events_comm_of_disjoint
    network e₁.1 e₂.1 e₁.2 e₂.2 hd X

/--
Swapping consecutive disjoint events anywhere within
a finite history preserves the resulting configuration.

The histories preceding and following the swapped
events may contain arbitrary admissible interactions.

The result follows from the history concatenation law
and commutativity of the two disjoint events.
-/
theorem execute_history_swap_disjoint_context
    {V : Type*} [DecidableEq V]
    (network : OscillatorNetwork V)
    (P S : EventHistory network)
    (e₁ e₂ : AdmissibleEvent network)
    (hd : DisjointAdmissibleEvents e₁ e₂)
    (X : OscillatorConfiguration V) :
    execute_history network (P ++ [e₁, e₂] ++ S) X =
      execute_history network (P ++ [e₂, e₁] ++ S) X := by
  repeat rw [execute_history_append]
  rw [execute_history_swap_disjoint network e₁ e₂ hd]


/-!
## Equivalence of Local Event Histories

Histories are related by elementary swaps of consecutive
disjoint events.

The resulting equivalence relation identifies histories
whose event ordering differs only by finitely many such
swaps.

No physical time coordinate or global clock is introduced.
-/

/--
Two histories differ by one admissible swap when they
have a common prefix and suffix, with two consecutive
disjoint events appearing in opposite orders.
-/
def HistorySwap
    {V : Type*}
    {network : OscillatorNetwork V}
    (H₁ H₂ : EventHistory network) : Prop :=
  ∃ (P S : EventHistory network)
    (e₁ e₂ : AdmissibleEvent network),
    DisjointAdmissibleEvents e₁ e₂ ∧
    H₁ = P ++ [e₁, e₂] ++ S ∧
    H₂ = P ++ [e₂, e₁] ++ S

/--
Two event histories are equivalent when they are connected
by a finite sequence of elementary disjoint-event swaps.

This is the reflexive, symmetric, and transitive closure
of `HistorySwap`.

Equivalent histories may differ in their computational
ordering, but only through exchanges of independent events.

No physical temporal interpretation is assumed.
-/
def HistoryEquivalent
    {V : Type*}
    {network : OscillatorNetwork V}
    (H₁ H₂ : EventHistory network) : Prop :=
  Relation.EqvGen
    (HistorySwap (network := network)) H₁ H₂

/--
Equivalent event histories produce identical final
oscillator configurations.

History equivalence is generated by swapping consecutive
disjoint admissible events, possibly within arbitrary
prefixes and suffixes.

The proof proceeds by induction over the equivalence
relation:

* An elementary swap preserves execution by
  `execute_history_swap_disjoint_context`.
* Reflexivity follows from equality.
* Symmetry reverses an established equality.
* Transitivity composes two established equalities.

Thus the result depends only on the history's equivalence
class, not on the particular ordering of independent events.

No physical time or duration is assumed.
-/
theorem execute_history_eq_of_equivalent
    {V : Type*} [DecidableEq V]
    (network : OscillatorNetwork V)
    (H₁ H₂ : EventHistory network)
    (h : HistoryEquivalent H₁ H₂)
    (X : OscillatorConfiguration V) :
    execute_history network H₁ X =
      execute_history network H₂ X := by
  unfold HistoryEquivalent at h
  induction h with
  | rel H₁ H₂ hs =>
      obtain ⟨P, S, e₁, e₂, hd, h₁, h₂⟩ := hs
      subst H₁
      subst H₂
      exact execute_history_swap_disjoint_context
        network P S e₁ e₂ hd X
  | refl H =>
      rfl
  | symm H₁ H₂ _ ih =>
      exact ih.symm
  | trans H₁ H₂ H₃ _ _ ih₁ ih₂ =>
      exact ih₁.trans ih₂

/--
Two admissible local events are structurally dependent
when their participating vertex sets overlap.

This is the negation of event disjointness.

Structural dependence means that exchanging the events
is not justified by the general disjoint-event
commutativity theorem.

It does not assert that either event actually changes
a shared state or transmits information.
-/
def DependentAdmissibleEvents
    {V : Type*}
    {network : OscillatorNetwork V}
    (e₁ e₂ : AdmissibleEvent network) : Prop :=
  ¬ DisjointAdmissibleEvents e₁ e₂

/--
Structural dependence between admissible events is symmetric.

If two events share a participating vertex, their overlap
does not depend on which event is considered first.
-/
theorem dependent_admissible_events_symm
    {V : Type*}
    {network : OscillatorNetwork V}
    (e₁ e₂ : AdmissibleEvent network)
    (h : DependentAdmissibleEvents e₁ e₂) :
    DependentAdmissibleEvents e₂ e₁ := by
  intro hd
  apply h
  rcases hd with ⟨h₁, h₂, h₃, h₄⟩
  exact ⟨Ne.symm h₁, Ne.symm h₃,
    Ne.symm h₂, Ne.symm h₄⟩


/-!
## Structural Dependencies Within Event Histories

Event occurrences are identified by their positions in a
finite history.

An earlier occurrence is directly related to a later
occurrence when their events share a participating vertex.

These relations record structural precedence within the
chosen history, not elapsed time or physical causation.
-/

/--
Two occurrences in a finite history have a direct
structural dependency when the first occurs earlier
in the list and their events share a participating vertex.

The indices identify occurrences, allowing the same
event value to appear multiple times in one history.

This relation does not assert that information was
actually transferred between the events.
-/
def HistoryDirectDependency
    {V : Type*}
    {network : OscillatorNetwork V}
    (H : EventHistory network)
    (i j : Fin H.length) : Prop :=
  i.val < j.val ∧
    DependentAdmissibleEvents (H.get i) (H.get j)

/--
No event occurrence directly precedes itself.

This follows from the strict ordering of occurrence
indices and does not depend on the event's interaction.
-/
theorem history_direct_dependency_irrefl
    {V : Type*}
    {network : OscillatorNetwork V}
    (H : EventHistory network)
    (i : Fin H.length) :
    ¬ HistoryDirectDependency H i i := by
  intro h
  exact (Nat.lt_irrefl i.val) h.1

/--
Direct structural dependency between event occurrences
is asymmetric.

If occurrence `i` directly precedes occurrence `j`,
then `j` cannot directly precede `i`.
-/
theorem history_direct_dependency_asymm
    {V : Type*}
    {network : OscillatorNetwork V}
    (H : EventHistory network)
    (i j : Fin H.length)
    (h : HistoryDirectDependency H i j) :
    ¬ HistoryDirectDependency H j i := by
  intro hji
  exact (Nat.lt_asymm h.1 hji.1)

/--
Structural precedence between event occurrences is the
transitive closure of direct structural dependency.

An occurrence precedes another when they are connected
by a nonempty chain of earlier, structurally dependent
occurrences.

This relation records dependency paths within a finite
history. It does not introduce physical time or assert
that information was actually transmitted.
-/
def HistoryPrecedes
    {V : Type*}
    {network : OscillatorNetwork V}
    (H : EventHistory network)
    (i j : Fin H.length) : Prop :=
  Relation.TransGen (HistoryDirectDependency H) i j

/--
Structural precedence is transitive.

If occurrence `i` precedes `j`, and `j` precedes `k`,
then `i` precedes `k`.
-/
theorem history_precedes_trans
    {V : Type*}
    {network : OscillatorNetwork V}
    (H : EventHistory network)
    (i j k : Fin H.length)
    (hij : HistoryPrecedes H i j)
    (hjk : HistoryPrecedes H j k) :
    HistoryPrecedes H i k := by
  exact Relation.TransGen.trans hij hjk

/--
Structural precedence respects the strict ordering
of event occurrences within the history.

Every direct dependency advances to a greater index,
so any nonempty chain of dependencies also advances
to a greater index.

The indices are bookkeeping positions, not physical
timestamps.
-/
theorem history_precedes_implies_lt
    {V : Type*}
    {network : OscillatorNetwork V}
    (H : EventHistory network)
    (i j : Fin H.length)
    (h : HistoryPrecedes H i j) :
    i.val < j.val := by
  induction h with
  | single h =>
      exact h.1
  | tail _ h ih =>
      exact lt_trans ih h.1

/--
No event occurrence structurally precedes itself.

A dependency chain must strictly increase occurrence
indices, making a cycle impossible.
-/
theorem history_precedes_irrefl
    {V : Type*}
    {network : OscillatorNetwork V}
    (H : EventHistory network)
    (i : Fin H.length) :
    ¬ HistoryPrecedes H i i := by
  intro h
  exact (Nat.lt_irrefl i.val)
    (history_precedes_implies_lt H i i h)

/--
An occurrence correspondence between two event histories
is a bijection between their occurrence indices that
preserves the event attached to each occurrence.

Occurrences are distinguished by position even when
their event values are equal.
-/
structure HistoryOccurrenceEquiv
    {V : Type*}
    {network : OscillatorNetwork V}
    (H₁ H₂ : EventHistory network) where
  indexEquiv : Fin H₁.length ≃ Fin H₂.length
  event_eq : ∀ i, H₁.get i = H₂.get (indexEquiv i)

/--
An occurrence correspondence preserves structural
precedence in both directions.

This describes an isomorphism of the labeled dependency
orders induced by the two histories.
-/
def PreservesHistoryPrecedence
    {V : Type*}
    {network : OscillatorNetwork V}
    {H₁ H₂ : EventHistory network}
    (f : HistoryOccurrenceEquiv H₁ H₂) : Prop :=
  ∀ i j,
    HistoryPrecedes H₁ i j ↔
      HistoryPrecedes H₂ (f.indexEquiv i) (f.indexEquiv j)

/--
The identity correspondence associates each event occurrence
with itself.
-/
def history_occurrence_equiv_refl
    {V : Type*}
    {network : OscillatorNetwork V}
    (H : EventHistory network) :
    HistoryOccurrenceEquiv H H where
  indexEquiv := Equiv.refl _
  event_eq := by
    intro i
    rfl

/--
The identity occurrence correspondence preserves
structural precedence.
-/
theorem history_occurrence_equiv_refl_preserves
    {V : Type*}
    {network : OscillatorNetwork V}
    (H : EventHistory network) :
    PreservesHistoryPrecedence
      (history_occurrence_equiv_refl H) := by
  intro i j
  rfl

/--
Reverse an occurrence correspondence using the inverse
bijection of occurrence indices.

Event labels remain unchanged under the correspondence.
-/
def HistoryOccurrenceEquiv.symm
    {V : Type*}
    {network : OscillatorNetwork V}
    {H₁ H₂ : EventHistory network}
    (f : HistoryOccurrenceEquiv H₁ H₂) :
    HistoryOccurrenceEquiv H₂ H₁ where
  indexEquiv := f.indexEquiv.symm
  event_eq := by
    intro j
    have h := f.event_eq (f.indexEquiv.symm j)
    simpa using h.symm

/--
Reversing a precedence-preserving occurrence correspondence
also preserves precedence.
-/
theorem history_occurrence_equiv_symm_preserves
    {V : Type*}
    {network : OscillatorNetwork V}
    {H₁ H₂ : EventHistory network}
    (f : HistoryOccurrenceEquiv H₁ H₂)
    (hf : PreservesHistoryPrecedence f) :
    PreservesHistoryPrecedence f.symm := by
  intro i j
  have h := hf (f.indexEquiv.symm i) (f.indexEquiv.symm j)
  change
    HistoryPrecedes H₂ i j ↔
      HistoryPrecedes H₁
        (f.indexEquiv.symm i)
        (f.indexEquiv.symm j)
  simpa only [Equiv.apply_symm_apply] using h.symm

/--
Compose two occurrence correspondences through an
intermediate history.

The resulting bijection preserves the event label
attached to every occurrence.
-/
def HistoryOccurrenceEquiv.trans
    {V : Type*}
    {network : OscillatorNetwork V}
    {H₁ H₂ H₃ : EventHistory network}
    (f : HistoryOccurrenceEquiv H₁ H₂)
    (g : HistoryOccurrenceEquiv H₂ H₃) :
    HistoryOccurrenceEquiv H₁ H₃ where
  indexEquiv := f.indexEquiv.trans g.indexEquiv
  event_eq := by
    intro i
    exact (f.event_eq i).trans
      (g.event_eq (f.indexEquiv i))

/--
The composition of precedence-preserving occurrence
correspondences also preserves precedence.
-/
theorem history_occurrence_equiv_trans_preserves
    {V : Type*}
    {network : OscillatorNetwork V}
    {H₁ H₂ H₃ : EventHistory network}
    (f : HistoryOccurrenceEquiv H₁ H₂)
    (g : HistoryOccurrenceEquiv H₂ H₃)
    (hf : PreservesHistoryPrecedence f)
    (hg : PreservesHistoryPrecedence g) :
    PreservesHistoryPrecedence (f.trans g) := by
  intro i j
  change
    HistoryPrecedes H₁ i j ↔
      HistoryPrecedes H₃
        (g.indexEquiv (f.indexEquiv i))
        (g.indexEquiv (f.indexEquiv j))
  exact (hf i j).trans
    (hg (f.indexEquiv i) (f.indexEquiv j))

/--
The permutation of occurrence indices induced by swapping
two consecutive events following a prefix of length `k`.

The hypothesis ensures that both positions belong to
the history.
-/
def adjacent_occurrence_swap
    (n k : ℕ)
    (h : k + 1 < n) :
    Fin n ≃ Fin n :=
  Equiv.swap
    (⟨k, by omega⟩ : Fin n)
    (⟨k + 1, h⟩ : Fin n)

/--
An adjacent occurrence swap is an involution.

Applying the same index swap twice restores the original
occurrence index.
-/
theorem adjacent_occurrence_swap_involutive
    (n k : ℕ)
    (h : k + 1 < n)
    (i : Fin n) :
    adjacent_occurrence_swap n k h
      (adjacent_occurrence_swap n k h i) = i := by
  unfold adjacent_occurrence_swap
  simp only [Equiv.swap_apply_self]

/--
An adjacent occurrence swap maps the first selected
position to the second.
-/
theorem adjacent_occurrence_swap_left
    (n k : ℕ)
    (h : k + 1 < n) :
    adjacent_occurrence_swap n k h
      (⟨k, by omega⟩ : Fin n) =
        (⟨k + 1, h⟩ : Fin n) := by
  unfold adjacent_occurrence_swap
  simp only [Equiv.swap_apply_left]

/--
An adjacent occurrence swap maps the second selected
position to the first.
-/
theorem adjacent_occurrence_swap_right
    (n k : ℕ)
    (h : k + 1 < n) :
    adjacent_occurrence_swap n k h
      (⟨k + 1, h⟩ : Fin n) =
        (⟨k, by omega⟩ : Fin n) := by
  unfold adjacent_occurrence_swap
  simp only [Equiv.swap_apply_right]

/--
An adjacent occurrence swap fixes every position other
than the two selected positions.

This property is independent of event values and of
the interactions associated with those occurrences.
-/
theorem adjacent_occurrence_swap_other
    (n k : ℕ)
    (h : k + 1 < n)
    (i : Fin n)
    (hi₁ : i.val ≠ k)
    (hi₂ : i.val ≠ k + 1) :
    adjacent_occurrence_swap n k h i = i := by
  unfold adjacent_occurrence_swap
  apply Equiv.swap_apply_of_ne_of_ne
  · intro heq
    exact hi₁ (congrArg Fin.val heq)
  · intro heq
    exact hi₂ (congrArg Fin.val heq)

/--
Swapping the first two elements of a list preserves
element lookup under the corresponding index exchange.

Positions zero and one are exchanged; all subsequent
positions are unchanged.

Optional indexing avoids requiring bounds proofs and
makes this lemma applicable to arbitrary element types.
-/
theorem getElem?_swap_first_two
    {α : Type*}
    (a b : α)
    (S : List α)
    (i : ℕ) :
    (a :: b :: S)[i]? =
      (b :: a :: S)[if i = 0 then 1
                     else if i = 1 then 0
                     else i]? := by
  cases i with
  | zero =>
      simp
  | succ j =>
      cases j with
      | zero =>
          simp
      | succ k =>
          simp

/--
Exchange natural-number indices `k` and `k + 1`,
leaving all other indices unchanged.

This is the unbounded-index counterpart of
`adjacent_occurrence_swap`.
-/
def adjacent_index (k i : ℕ) : ℕ :=
  if i = k then k + 1
  else if i = k + 1 then k
  else i

/--
When the exchanged positions are both strictly positive,
the index zero remains unchanged.
-/
theorem adjacent_index_zero (k : ℕ) :
    adjacent_index (k + 1) 0 = 0 := by
  unfold adjacent_index
  split_ifs <;> omega

/--
Moving the swap one position to the right commutes
with shifting an occurrence index by one.
-/
theorem adjacent_index_succ (k i : ℕ) :
    adjacent_index (k + 1) (i + 1) =
      adjacent_index k i + 1 := by
  unfold adjacent_index
  split_ifs <;> omega

/--
Swapping two consecutive elements after an arbitrary
prefix preserves optional element lookup under the
corresponding adjacent index permutation.

The swap occurs at positions `P.length` and
`P.length + 1`.

This result is independent of the element type and
requires no assumptions about the exchanged elements.
-/
theorem getElem?_swap_after_prefix
    {α : Type*}
    (P : List α)
    (a b : α)
    (S : List α)
    (i : ℕ) :
    (P ++ (a :: b :: S))[i]? =
      (P ++ (b :: a :: S))[adjacent_index P.length i]? := by
  induction P generalizing i with
  | nil =>
      simpa [adjacent_index] using
        getElem?_swap_first_two a b S i
  | cons x P ih =>
      cases i with
      | zero =>
          simp [adjacent_index_zero]
      | succ j =>
          simpa [adjacent_index_succ] using ih j

/--
The finite adjacent-occurrence swap agrees with the
corresponding natural-number index permutation.

Both representations exchange positions `k` and `k + 1`
and leave all other positions unchanged.

This connects the generic list-indexing lemmas to the
bounded occurrence indices used by event histories.
-/
theorem adjacent_occurrence_swap_val
    (n k : ℕ)
    (h : k + 1 < n)
    (i : Fin n) :
    (adjacent_occurrence_swap n k h i).val =
      adjacent_index k i.val := by
  by_cases hi₁ : i.val = k
  · have hi : i = (⟨k, by omega⟩ : Fin n) :=
      Fin.ext hi₁
    rw [hi, adjacent_occurrence_swap_left]
    simp [adjacent_index]
  · by_cases hi₂ : i.val = k + 1
    · have hi : i = (⟨k + 1, h⟩ : Fin n) :=
        Fin.ext hi₂
      rw [hi, adjacent_occurrence_swap_right]
      simp [adjacent_index]
    · rw [adjacent_occurrence_swap_other n k h i hi₁ hi₂]
      simp [adjacent_index, hi₁, hi₂]

/--
Swapping two consecutive elements after an arbitrary prefix
preserves optional lookup under the corresponding finite
occurrence permutation.

This connects the generic list-lookup theorem with the
bounded indices used to identify event occurrences.

The result is independent of element values and requires
no assumption that the exchanged elements are disjoint.
-/
theorem getElem?_swap_after_prefix_fin
    {α : Type*}
    (P : List α)
    (a b : α)
    (S : List α)
    (h : P.length + 1 <
      (P ++ (a :: b :: S)).length)
    (i : Fin (P ++ (a :: b :: S)).length) :
    (P ++ (a :: b :: S))[i.val]? =
      (P ++ (b :: a :: S))[(adjacent_occurrence_swap
        (P ++ (a :: b :: S)).length
        P.length h i).val]? := by
  rw [adjacent_occurrence_swap_val]
  exact getElem?_swap_after_prefix P a b S i.val

/--
The occurrence-index equivalence between lists that
differ by exchanging two consecutive elements after
an arbitrary prefix.

The adjacent swap exchanges the selected positions.
The final cast accounts for the equality of the two
list lengths.

No assumptions about the element values are required.
-/
def swap_after_prefix_index_equiv
    {α : Type*}
    (P : List α)
    (a b : α)
    (S : List α) :
    Fin (P ++ (a :: b :: S)).length ≃
      Fin (P ++ (b :: a :: S)).length := by
  have hlen :
      (P ++ (a :: b :: S)).length =
        (P ++ (b :: a :: S)).length := by
    simp
  have hk :
      P.length + 1 < (P ++ (a :: b :: S)).length := by
    simp only [List.length_append, List.length_cons]
    omega
  exact
    (adjacent_occurrence_swap
      (P ++ (a :: b :: S)).length
      P.length hk).trans
      (Fin.castOrderIso hlen).toEquiv

/--
The occurrence-index equivalence agrees with the
natural-number adjacent-index permutation.

Casting between equal finite-index types does not
change the underlying natural-number index.
-/
theorem swap_after_prefix_index_equiv_val
    {α : Type*}
    (P : List α)
    (a b : α)
    (S : List α)
    (i : Fin (P ++ (a :: b :: S)).length) :
    (swap_after_prefix_index_equiv P a b S i).val =
      adjacent_index P.length i.val := by
  unfold swap_after_prefix_index_equiv
  change
    ((Fin.castOrderIso _)
      (adjacent_occurrence_swap _ _ _ i)).val =
      adjacent_index P.length i.val
  rw [Fin.coe_orderIso_apply]
  exact adjacent_occurrence_swap_val _ _ _ i

/--
The occurrence-index equivalence for an adjacent swap
preserves the element associated with each occurrence.

The two lists differ only by exchanging consecutive
elements after a common prefix.

The result applies to arbitrary element types and
does not require the exchanged elements to be distinct.
-/
theorem get_swap_after_prefix
    {α : Type*}
    (P : List α)
    (a b : α)
    (S : List α)
    (i : Fin (P ++ (a :: b :: S)).length) :
    (P ++ (a :: b :: S)).get i =
      (P ++ (b :: a :: S)).get
        (swap_after_prefix_index_equiv P a b S i) := by
  have hlookup :
      (P ++ (a :: b :: S))[i.val]? =
        (P ++ (b :: a :: S))[
          (swap_after_prefix_index_equiv P a b S i).val]? := by
    rw [swap_after_prefix_index_equiv_val]
    exact getElem?_swap_after_prefix P a b S i.val
  rw [List.getElem?_eq_getElem i.isLt,
      List.getElem?_eq_getElem
        (swap_after_prefix_index_equiv P a b S i).isLt] at hlookup
  exact Option.some.inj
    (by simpa only [List.get_eq_getElem] using hlookup)

/--
Construct the occurrence correspondence induced by
exchanging two consecutive events within a history.

The correspondence exchanges their occurrence indices
and preserves the event label at every position.

No disjointness hypothesis is required: this construction
establishes occurrence correspondence, not preservation
of structural precedence.
-/
def history_occurrence_equiv_swap
    {V : Type*}
    {network : OscillatorNetwork V}
    (P S : EventHistory network)
    (e₁ e₂ : AdmissibleEvent network) :
    HistoryOccurrenceEquiv
      (P ++ (e₁ :: e₂ :: S))
      (P ++ (e₂ :: e₁ :: S)) where
  indexEquiv :=
    swap_after_prefix_index_equiv P e₁ e₂ S
  event_eq := by
    intro i
    exact get_swap_after_prefix P e₁ e₂ S i

/--
An adjacent index swap preserves strict ordering between
two indices, except when they are exactly the exchanged
pair in their original order.

This is a purely arithmetic property of the permutation.
It does not depend on event values or network structure.
-/
theorem adjacent_index_lt_of_lt_of_not_pair
    (k i j : ℕ)
    (hij : i < j)
    (hnot : ¬ (i = k ∧ j = k + 1)) :
    adjacent_index k i < adjacent_index k j := by
  unfold adjacent_index
  split_ifs <;> omega

/--
An adjacent index swap preserves strict ordering in both
directions, provided the two indices are not exactly the
exchanged pair in either orientation.

The only ordering reversal introduced by an adjacent
swap is between the two exchanged positions themselves.

This is an arithmetic property, independent of event
values and network structure.
-/
theorem adjacent_index_lt_iff_of_not_swapped_pair
    (k i j : ℕ)
    (h₁ : ¬ (i = k ∧ j = k + 1))
    (h₂ : ¬ (i = k + 1 ∧ j = k)) :
    adjacent_index k i < adjacent_index k j ↔
      i < j := by
  constructor
  · intro h
    unfold adjacent_index at h
    split_ifs at h <;> omega
  · intro h
    exact adjacent_index_lt_of_lt_of_not_pair
      k i j h h₁

/--
Disjoint admissible events are not structurally dependent
in either orientation.

This excludes the exchanged pair from the dependency
relation when constructing an occurrence correspondence
for an adjacent swap.

The result depends only on shared vertex participation,
not on the interactions themselves or any geometry.
-/
theorem disjoint_admissible_events_not_dependent
    {V : Type*}
    {network : OscillatorNetwork V}
    (e₁ e₂ : AdmissibleEvent network)
    (hd : DisjointAdmissibleEvents e₁ e₂) :
    ¬ DependentAdmissibleEvents e₁ e₂ ∧
    ¬ DependentAdmissibleEvents e₂ e₁ := by
  constructor
  · intro h
    exact h hd
  · intro h
    exact (dependent_admissible_events_symm e₂ e₁ h) hd

/--
The two positions immediately following a prefix contain
the first and second elements of the appended pair.

Optional indexing permits this statement without
explicit finite-index bounds.
-/
theorem getElem?_pair_after_prefix
    {α : Type*}
    (P : List α)
    (a b : α)
    (S : List α) :
    (P ++ (a :: b :: S))[P.length]? = some a ∧
    (P ++ (a :: b :: S))[P.length + 1]? = some b := by
  induction P with
  | nil =>
      simp
  | cons x P ih =>
      change
        (P ++ (a :: b :: S))[P.length]? = some a ∧
        (P ++ (a :: b :: S))[P.length + 1]? = some b
      exact ih

/--
The event at either selected occurrence index is the
corresponding element of the pair following the prefix.

This converts optional list lookup into ordinary
finite-index lookup.
-/
theorem get_pair_after_prefix
    {α : Type*}
    (P : List α)
    (a b : α)
    (S : List α)
    (i : Fin (P ++ (a :: b :: S)).length) :
    (i.val = P.length →
      (P ++ (a :: b :: S)).get i = a) ∧
    (i.val = P.length + 1 →
      (P ++ (a :: b :: S)).get i = b) := by
  constructor
  · intro hi
    have hlookup :
        (P ++ (a :: b :: S))[i.val]? = some a := by
      rw [hi]
      exact (getElem?_pair_after_prefix P a b S).1
    rw [List.getElem?_eq_getElem i.isLt] at hlookup
    exact Option.some.inj
      (by simpa only [List.get_eq_getElem] using hlookup)
  · intro hi
    have hlookup :
        (P ++ (a :: b :: S))[i.val]? = some b := by
      rw [hi]
      exact (getElem?_pair_after_prefix P a b S).2
    rw [List.getElem?_eq_getElem i.isLt] at hlookup
    exact Option.some.inj
      (by simpa only [List.get_eq_getElem] using hlookup)

/--
Structurally dependent occurrences cannot be exactly the
two positions occupied by consecutive disjoint events,
in either orientation.

This excludes the only pair of indices whose relative
ordering is reversed by an adjacent occurrence swap.

The result uses relational event disjointness and makes
no geometric assumptions.
-/
theorem dependent_occurrences_not_swapped_pair
    {V : Type*}
    {network : OscillatorNetwork V}
    (P S : EventHistory network)
    (e₁ e₂ : AdmissibleEvent network)
    (hd : DisjointAdmissibleEvents e₁ e₂)
    (i j : Fin (P ++ (e₁ :: e₂ :: S)).length)
    (hdep : DependentAdmissibleEvents
      ((P ++ (e₁ :: e₂ :: S)).get i)
      ((P ++ (e₁ :: e₂ :: S)).get j)) :
    ¬ (i.val = P.length ∧ j.val = P.length + 1) ∧
    ¬ (i.val = P.length + 1 ∧ j.val = P.length) := by
  obtain ⟨h₁₂, h₂₁⟩ :=
    disjoint_admissible_events_not_dependent e₁ e₂ hd
  constructor
  · rintro ⟨hi, hj⟩
    have h_i :=
      (get_pair_after_prefix P e₁ e₂ S i).1 hi
    have h_j :=
      (get_pair_after_prefix P e₁ e₂ S j).2 hj
    exact h₁₂ (by simpa only [h_i, h_j] using hdep)
  · rintro ⟨hi, hj⟩
    have h_i :=
      (get_pair_after_prefix P e₁ e₂ S i).2 hi
    have h_j :=
      (get_pair_after_prefix P e₁ e₂ S j).1 hj
    exact h₂₁ (by simpa only [h_i, h_j] using hdep)

/--
Swapping two consecutive disjoint admissible events
preserves direct structural dependency under the
corresponding occurrence-index equivalence.

The proof combines event-label preservation with the
fact that the only reversed index pair consists of
structurally independent events.

Thus direct dependency is preserved in both directions,
without assuming physical time or geometric proximity.
-/
theorem history_direct_dependency_swap_iff
    {V : Type*}
    {network : OscillatorNetwork V}
    (P S : EventHistory network)
    (e₁ e₂ : AdmissibleEvent network)
    (hd : DisjointAdmissibleEvents e₁ e₂)
    (i j : Fin (P ++ (e₁ :: e₂ :: S)).length) :
    HistoryDirectDependency
        (P ++ (e₁ :: e₂ :: S)) i j ↔
      HistoryDirectDependency
        (P ++ (e₂ :: e₁ :: S))
        ((history_occurrence_equiv_swap P S e₁ e₂).indexEquiv i)
        ((history_occurrence_equiv_swap P S e₁ e₂).indexEquiv j) := by
  let f := history_occurrence_equiv_swap P S e₁ e₂
  have hi :
      (P ++ (e₁ :: e₂ :: S)).get i =
        (P ++ (e₂ :: e₁ :: S)).get (f.indexEquiv i) :=
    f.event_eq i
  have hj :
      (P ++ (e₁ :: e₂ :: S)).get j =
        (P ++ (e₂ :: e₁ :: S)).get (f.indexEquiv j) :=
    f.event_eq j
  have hvi :
      (f.indexEquiv i).val = adjacent_index P.length i.val :=
    swap_after_prefix_index_equiv_val P e₁ e₂ S i
  have hvj :
      (f.indexEquiv j).val = adjacent_index P.length j.val :=
    swap_after_prefix_index_equiv_val P e₁ e₂ S j
  change
    (i.val < j.val ∧
      DependentAdmissibleEvents
        ((P ++ (e₁ :: e₂ :: S)).get i)
        ((P ++ (e₁ :: e₂ :: S)).get j)) ↔
    ((f.indexEquiv i).val < (f.indexEquiv j).val ∧
      DependentAdmissibleEvents
        ((P ++ (e₂ :: e₁ :: S)).get (f.indexEquiv i))
        ((P ++ (e₂ :: e₁ :: S)).get (f.indexEquiv j)))
  constructor
  · rintro ⟨hord, hdep⟩
    obtain ⟨hnot₁, hnot₂⟩ :=
      dependent_occurrences_not_swapped_pair
        P S e₁ e₂ hd i j hdep
    constructor
    · rw [hvi, hvj]
      exact
        (adjacent_index_lt_iff_of_not_swapped_pair
          P.length i.val j.val hnot₁ hnot₂).2 hord
    · simpa only [hi, hj] using hdep
  · rintro ⟨hord, hdep₂⟩
    have hdep₁ :
        DependentAdmissibleEvents
          ((P ++ (e₁ :: e₂ :: S)).get i)
          ((P ++ (e₁ :: e₂ :: S)).get j) := by
      simpa only [hi, hj] using hdep₂
    obtain ⟨hnot₁, hnot₂⟩ :=
      dependent_occurrences_not_swapped_pair
        P S e₁ e₂ hd i j hdep₁
    constructor
    · rw [hvi, hvj] at hord
      exact
        (adjacent_index_lt_iff_of_not_swapped_pair
          P.length i.val j.val hnot₁ hnot₂).1 hord
    · exact hdep₁

/--
An equivalence that preserves a binary relation in both
directions also preserves its transitive closure.

Every finite chain of related elements is transported
through the equivalence, and the inverse equivalence
transports chains in the opposite direction.

This result is independent of oscillator networks,
event histories, and geometric interpretation.
-/
theorem transGen_iff_of_equiv
    {α β : Type*}
    (r : α → α → Prop)
    (s : β → β → Prop)
    (f : α ≃ β)
    (hrel : ∀ a b, r a b ↔ s (f a) (f b))
    (a b : α) :
    Relation.TransGen r a b ↔
      Relation.TransGen s (f a) (f b) := by
  constructor
  · intro h
    induction h with
    | single h =>
        exact Relation.TransGen.single
          ((hrel _ _).1 h)
    | tail _ h ih =>
        exact Relation.TransGen.tail ih
          ((hrel _ _).1 h)
  · intro h
    have hrel_inv (x y : β) :
        s x y ↔ r (f.symm x) (f.symm y) := by
      simpa only [Equiv.apply_symm_apply] using
        (hrel (f.symm x) (f.symm y)).symm
    have hback :
        ∀ {x y : β}, Relation.TransGen s x y →
          Relation.TransGen r (f.symm x) (f.symm y) := by
      intro x y hxy
      induction hxy with
      | single hs =>
          exact Relation.TransGen.single
            ((hrel_inv _ _).1 hs)
      | tail _ hs ih =>
          exact Relation.TransGen.tail ih
            ((hrel_inv _ _).1 hs)
    simpa only [Equiv.symm_apply_apply] using (hback h)

/--
Swapping two consecutive disjoint admissible events
preserves structural precedence between all corresponding
event occurrences.

The result follows from preservation of direct structural
dependency and the invariance of transitive closure under
a relation-preserving equivalence.

No physical time, spatial distance, or metric is assumed.
-/
theorem history_precedes_swap_iff
    {V : Type*}
    {network : OscillatorNetwork V}
    (P S : EventHistory network)
    (e₁ e₂ : AdmissibleEvent network)
    (hd : DisjointAdmissibleEvents e₁ e₂)
    (i j : Fin (P ++ (e₁ :: e₂ :: S)).length) :
    HistoryPrecedes
        (P ++ (e₁ :: e₂ :: S)) i j ↔
      HistoryPrecedes
        (P ++ (e₂ :: e₁ :: S))
        ((history_occurrence_equiv_swap P S e₁ e₂).indexEquiv i)
        ((history_occurrence_equiv_swap P S e₁ e₂).indexEquiv j) := by
  let f := history_occurrence_equiv_swap P S e₁ e₂
  change
    Relation.TransGen
      (HistoryDirectDependency (P ++ (e₁ :: e₂ :: S))) i j ↔
    Relation.TransGen
      (HistoryDirectDependency (P ++ (e₂ :: e₁ :: S)))
      (f.indexEquiv i) (f.indexEquiv j)
  exact transGen_iff_of_equiv
    (HistoryDirectDependency (P ++ (e₁ :: e₂ :: S)))
    (HistoryDirectDependency (P ++ (e₂ :: e₁ :: S)))
    f.indexEquiv
    (fun x y =>
      history_direct_dependency_swap_iff
        P S e₁ e₂ hd x y)
    i j

/--
The occurrence correspondence induced by swapping two
consecutive disjoint admissible events preserves
structural precedence in both directions.

The correspondence preserves event labels, while
disjointness guarantees that the exchanged occurrences
have no direct structural dependency.

Consequently, the transitive dependency order is
preserved under the correspondence.

No physical time, metric, or geometric proximity
is assumed.
-/
theorem history_occurrence_equiv_swap_preserves
    {V : Type*}
    {network : OscillatorNetwork V}
    (P S : EventHistory network)
    (e₁ e₂ : AdmissibleEvent network)
    (hd : DisjointAdmissibleEvents e₁ e₂) :
    PreservesHistoryPrecedence
      (history_occurrence_equiv_swap P S e₁ e₂) := by
  intro i j
  exact history_precedes_swap_iff P S e₁ e₂ hd i j

/--
Histories related by one elementary disjoint-event swap
have isomorphic labeled structural dependency orders.

The correspondence preserves event labels and structural
precedence in both directions.

This theorem packages the elementary swap construction
for use with the history-equivalence relation.
-/
theorem history_swap_has_precedence_equiv
    {V : Type*}
    {network : OscillatorNetwork V}
    (H₁ H₂ : EventHistory network)
    (hs : HistorySwap H₁ H₂) :
    ∃ f : HistoryOccurrenceEquiv H₁ H₂,
      PreservesHistoryPrecedence f := by
  rcases hs with ⟨P, S, e₁, e₂, hd, h₁, h₂⟩
  have hleft :
      H₁ = P ++ (e₁ :: e₂ :: S) := by
    calc
      H₁ = P ++ [e₁, e₂] ++ S := h₁
      _ = P ++ (e₁ :: e₂ :: S) := by
        simp only [List.append_assoc,
          List.cons_append, List.nil_append]
  have hright :
      H₂ = P ++ (e₂ :: e₁ :: S) := by
    calc
      H₂ = P ++ [e₂, e₁] ++ S := h₂
      _ = P ++ (e₂ :: e₁ :: S) := by
        simp only [List.append_assoc,
          List.cons_append, List.nil_append]
  rw [hleft, hright]
  exact ⟨history_occurrence_equiv_swap P S e₁ e₂,
    history_occurrence_equiv_swap_preserves
      P S e₁ e₂ hd⟩

/--
Equivalent finite event histories have isomorphic labeled
structural dependency partial orders.

The isomorphism associates occurrences of identical events
and preserves structural precedence in both directions.

The result follows by induction over history equivalence:

* An elementary swap preserves precedence.
* The identity correspondence preserves precedence.
* Inverting a correspondence preserves precedence.
* Composing correspondences preserves precedence.

Thus the dependency structure is invariant under finite
reorderings generated by swaps of independent events.

No physical time, geometric distance, metric, or
propagation speed is assumed.
-/
theorem history_equivalent_has_precedence_equiv
    {V : Type*}
    {network : OscillatorNetwork V}
    (H₁ H₂ : EventHistory network)
    (heq : HistoryEquivalent H₁ H₂) :
    ∃ f : HistoryOccurrenceEquiv H₁ H₂,
      PreservesHistoryPrecedence f := by
  unfold HistoryEquivalent at heq
  induction heq with
  | rel H₁ H₂ hs =>
      exact history_swap_has_precedence_equiv H₁ H₂ hs
  | refl H =>
      exact ⟨history_occurrence_equiv_refl H,
        history_occurrence_equiv_refl_preserves H⟩
  | symm H₁ H₂ _ ih =>
      obtain ⟨f, hf⟩ := ih
      exact ⟨f.symm,
        history_occurrence_equiv_symm_preserves f hf⟩
  | trans H₁ H₂ H₃ _ _ ih₁ ih₂ =>
      obtain ⟨f, hf⟩ := ih₁
      obtain ⟨g, hg⟩ := ih₂
      exact ⟨f.trans g,
        history_occurrence_equiv_trans_preserves
          f g hf hg⟩


end StandingWave
