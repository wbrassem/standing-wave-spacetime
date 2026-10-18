/-
Copyright (c) 2026 Wayne Brassem. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Wayne Brassem
-/

import StandingWave.Oscillator

/-!
# Pairwise Oscillator Coupling and Magnitude Invariants

This module extends the primitive oscillator framework to
interactions between pairs of abstract oscillator states.

Each oscillator retains the two-component structure
introduced in `StandingWave.Oscillator`. A combined
quadratic magnitude is defined for pairs of states, and
general interactions are represented as transformations
from pairs of oscillator states to pairs of oscillator
states.

The objective is to investigate conservation, redistribution,
and loss of quadratic magnitude under such interactions,
and to establish the algebraic structures associated with
their composition.

No physical energy, force, propagation mechanism, spatial
geometry, or temporal evolution law is assumed.

The mathematical properties of interactions are established
independently of any proposed physical interpretation.

## Foundational Principles

### Interaction Without Physical Assumptions

An oscillator interaction is an arbitrary transformation

    OscillatorState × OscillatorState →
      OscillatorState × OscillatorState.

This definition does not require an interaction to be
linear, continuous, conservative, reversible, or physically
realizable.

Such properties must be established separately for
particular interactions or classes of interactions.

The ordering of inputs and outputs is part of the
mathematical representation. It does not itself establish
a preferred direction in physical space.

Similarly, composition specifies an order of function
application without introducing a physical time parameter.

### Quadratic Magnitude Is Not Presumed to Be Energy

For an oscillator state A = (q_A, p_A), the primitive
quadratic magnitude is

    Q(A) = q_A² + p_A².

For two states A and B, the combined magnitude is

    Q₂(A, B) = Q(A) + Q(B).

This is a nonnegative mathematical quantity.

Although the terminology of conservation, transfer,
and loss is useful for describing changes in Q₂,
no identification with physical energy is assumed.

In particular, the conservation of quadratic magnitude
does not establish energy conservation, and a decrease
in quadratic magnitude does not by itself establish
dissipation, irreversibility, or information loss.

### Conservation Is a Property, Not a Premise

The general interaction type imposes no conservation law.

Instead, conservation and non-increase are expressed
through explicit predicates.

An interaction is conservative when it preserves Q₂
for every possible input pair.

An interaction is non-increasing when its output Q₂
never exceeds its input Q₂.

Strict magnitude loss requires a decrease for at least
one input pair.

These distinctions allow interactions with different
mathematical behaviors to coexist in the framework.

## Mathematical Framework

### Combined Quadratic Magnitude

The combined magnitude of two oscillator states is

    Q₂(A, B) = Q(A) + Q(B).

The module establishes:

    Q₂(A, B) ≥ 0,

    Q₂(A, B) = 0
      ↔ A = (0, 0) ∧ B = (0, 0),

and

    Q₂(A, B) = Q₂(B, A).

Each individual quadratic magnitude is bounded above
by the combined magnitude.

These properties follow from the positive definiteness
and nonnegativity established in `Oscillator.lean`.

### Normalized Pairwise Mixing

A family of pairwise interactions is defined using
normalized real coefficients c and s satisfying

    c² + s² = 1.

The mixing transformation is

    A' = c*A - s*B,
    B' = s*A + c*B,

where the operations act componentwise on the two
oscillator states.

Although the coefficients belong to `NormalizedRotation`,
this transformation mixes the two oscillator states
rather than rotating the internal coordinates of one
state.

The module proves that normalized mixing preserves
combined quadratic magnitude:

    Q₂(A', B') = Q₂(A, B).

Individual oscillator magnitudes need not be conserved.

Consequently, normalized mixing permits redistribution
of quadratic magnitude between participating states
while preserving their combined magnitude.

Concrete examples demonstrate both complete and
partial redistribution.

### Inner Products and Magnitude Redistribution

The Euclidean inner product of two oscillator states
is defined by

    <A, B> = q_A*q_B + p_A*p_B.

The module establishes

    <A, A> = Q(A).

For normalized mixing, the individual output magnitudes
have the exact forms

    Q(A') =
      c²*Q(A) + s²*Q(B) - 2*c*s*<A, B>,

    Q(B') =
      s²*Q(A) + c²*Q(B) + 2*c*s*<A, B>.

The cross terms have opposite signs and cancel when
the two output magnitudes are added.

Thus the distribution of quadratic magnitude depends
not only on the individual input magnitudes but also
on the inner product between the states.

The module also establishes the two-dimensional
Lagrange identity:

    Q(A)*Q(B) - <A, B>²
      = (q_A*p_B - p_A*q_B)².

Nonnegativity of the squared determinant yields the
Cauchy-Schwarz inequality:

    <A, B>² ≤ Q(A)*Q(B).

These identities are algebraic properties of the
two-component oscillator states and require no
physical interpretation.

### Magnitude Change and Non-Increasing Interactions

The net magnitude change associated with an interaction
F and input states A and B is

    ΔQ₂(F, A, B) = Q₂(F(A, B)) - Q₂(A, B),

where Q₂ is applied to the two components of the
resulting pair.

Its sign distinguishes three possibilities:

* Negative change: strict decrease in combined magnitude.
* Zero change: conservation for that input.
* Positive change: increase in combined magnitude.

An interaction is conservative precisely when its
magnitude change vanishes for every input pair.

Every conservative interaction is non-increasing,
but the converse need not hold.

Uniform scaling provides a concrete family of
interactions exhibiting these distinctions.

For a real coefficient k, scaling both oscillator
states gives

    Q₂(k*A, k*B) = k²*Q₂(A, B).

If k² ≤ 1, the interaction is non-increasing.

Scaling by one half provides an explicit witness of
strict magnitude loss, demonstrating that non-increase
does not imply conservation.

No physical dissipation mechanism is inferred from
this mathematical decrease.

### Composition of Interactions

Interactions are composed as functions on pairs of
oscillator states.

The operation

    compose_interactions F G

applies G first, followed by F.

The module establishes that:

* Composition of conservative interactions is
  conservative.

* Composition of non-increasing interactions is
  non-increasing.

* Composition of a conservative interaction with a
  non-increasing interaction is non-increasing in
  either order.

* Interaction composition is associative.

An identity interaction returns its input pair
unchanged and acts as a two-sided identity.

These results establish that oscillator interactions
form a monoid under composition.

Unlike the normalized rotation group, this monoid
does not require its elements to possess inverses.

Nor is interaction composition assumed to be
commutative.

### Conservative and Non-Increasing Submonoids

Two submonoids of the interaction monoid are defined.

The conservative submonoid contains precisely those
interactions satisfying

    ConservesCombinedMagnitude F.

The non-increasing submonoid contains precisely those
interactions satisfying

    NonIncreasingCombinedMagnitude F.

Both contain the identity interaction and are closed
under composition.

Every conservative interaction is non-increasing,
giving the inclusion

    conservative_interactions
      ≤ nonincreasing_interactions.

The module establishes that this inclusion is strict:

    conservative_interactions
      < nonincreasing_interactions.

The proof uses uniform scaling by one half as a
non-increasing interaction that exhibits strict
magnitude loss and therefore cannot be conservative.

Thus the distinction between conservation and
non-increase is reflected in a proper inclusion
between algebraic structures.

## Principal Results

* `combined_magnitude_sq_nonneg`:
  Nonnegativity of the combined quadratic magnitude.

* `combined_magnitude_sq_eq_zero_iff`:
  The combined magnitude vanishes exactly when both
  oscillator states are zero.

* `combined_magnitude_sq_comm`:
  Invariance of combined magnitude under exchange
  of the two oscillator states.

* `mix_oscillators_conserves`:
  Conservation of combined magnitude under normalized
  pairwise mixing.

* `oscillator_inner_product_self`:
  The inner product of a state with itself equals
  its quadratic magnitude.

* `mix_oscillators_subtractive_output_magnitude`:
  Exact quadratic-magnitude formula for the subtractive
  mixing output `c*A - s*B`, including its negative
  algebraic cross term.

* `mix_oscillators_additive_output_magnitude`:
  Exact quadratic-magnitude formula for the additive
  mixing output `s*A + c*B`, including its positive
  algebraic cross term.

* `oscillator_lagrange_identity`:
  The two-dimensional Lagrange identity.

* `oscillator_inner_product_sq_le`:
  The Cauchy-Schwarz inequality for oscillator states.

* `conserves_iff_zero_magnitude_change`:
  Conservation is equivalent to zero net magnitude
  change for every input pair.

* `conserves_implies_nonincreasing`:
  Every conservative interaction is non-increasing.

* `scale_oscillators_magnitude`:
  Quadratic scaling law for combined magnitude.

* `scale_oscillators_nonincreasing`:
  Non-increase under scaling when k² ≤ 1.

* `scale_half_exhibits_loss`:
  Explicit existence of strict magnitude loss.

* `compose_interactions_conserves`:
  Closure of conservative interactions under composition.

* `compose_interactions_nonincreasing`:
  Closure of non-increasing interactions under composition.

* `compose_interactions_assoc`:
  Associativity of interaction composition.

* `identity_interaction_conserves`:
  Conservation under the identity interaction.

* `conservative_interactions_le_nonincreasing`:
  Inclusion of conservative interactions in the
  non-increasing submonoid.

* `conservative_interactions_lt_nonincreasing`:
  Proper inclusion of conservative interactions in
  the non-increasing submonoid.

## Development Structure

The principal constructions are organized into four
related developments.

1. Combined magnitude and conservative mixing

    OscillatorState
      |
      +-- oscillator_magnitude_sq
            |
            +-- combined_magnitude_sq
                  |
                  +-- Nonnegativity
                  |
                  +-- Positive definiteness
                  |
                  +-- OscillatorInteraction
                        |
                        +-- ConservesCombinedMagnitude
                        |
                        +-- mix_oscillators
                              |
                              +-- mix_oscillators_conserves

2. Inner products and magnitude changes

    oscillator_inner_product
      |
      +-- mix_oscillators_subtractive_output_magnitude
      |
      +-- mix_oscillators_additive_output_magnitude
      |
      +-- oscillator_lagrange_identity
            |
            +-- oscillator_inner_product_sq_le

    interaction_magnitude_change
      |
      +-- ConservesCombinedMagnitude
      |
      +-- NonIncreasingCombinedMagnitude
      |
      +-- ExhibitsMagnitudeLoss
      |
      +-- scale_oscillators
            |
            +-- scale_oscillators_magnitude
            |
            +-- scale_oscillators_nonincreasing
            |
            +-- scale_half_exhibits_loss

3. Composition and monoid structure

    OscillatorInteraction
      |
      +-- compose_interactions
      |     |
      |     +-- Preservation of conservation
      |     |
      |     +-- Preservation of non-increase
      |     |
      |     +-- Associativity
      |
      +-- identity_interaction
            |
            +-- Identity laws
                  |
                  +-- Monoid OscillatorInteraction

4. Submonoids and their relationship

    Monoid OscillatorInteraction
      |
      +-- conservative_interactions
      |     |
      |     +-- Identity and composition closure
      |
      +-- nonincreasing_interactions
            |
            +-- Identity and composition closure
            |
            +-- conservative_interactions_le_nonincreasing
            |
            +-- conservative_interactions_lt_nonincreasing

These diagrams summarize the mathematical organization
of the development rather than the exact dependencies
of every Lean declaration.

## Scope and Interpretation

This module establishes algebraic properties of
pairwise transformations of abstract oscillator states.

The combined quadratic magnitude is a mathematical
invariant for certain interactions, not an assumed
physical energy or other observable quantity.

Normalized mixing demonstrates that individual
quadratic magnitudes may change while their sum
remains conserved.

Uniform scaling demonstrates that a mathematically
well-defined interaction need not conserve the
combined magnitude.

Neither result establishes a physical mechanism
for energy exchange or dissipation.

A reduction in quadratic magnitude does not
necessarily imply irreversibility, just as conservation
of quadratic magnitude does not necessarily imply
invertibility.

Interaction composition describes successive function
application, not a prescribed evolution through
physical time.

Likewise, this module does not establish a spatial
relationship between interacting oscillator states.

No network adjacency, geometric proximity, distance,
propagation speed, or locality constraint is imposed
on the general `OscillatorInteraction` type.

The separate `Locality` development introduces an
admissibility relation governing which vertices may
participate in direct pairwise interactions.

That notion of locality is relational rather than
geometric and imposes additional structure without
changing the pairwise algebra developed here.

The present module therefore supplies mathematical
interaction laws and invariant classes that can be
used in subsequent network-level investigations.

Their physical significance, if any, remains a
question for further mathematical development and
empirical investigation.
-/

namespace StandingWave

/-!
## Combined Quadratic Magnitude

Define the combined quadratic magnitude of two oscillator
states and establish its elementary properties.

These include nonnegativity, positive definiteness,
symmetry under exchange of the two states, and bounds
on the individual magnitudes.

No interaction or conservation law is assumed.
-/

/--
The combined quadratic magnitude of two oscillator states.

The total is defined additively as the sum of the
individual quadratic magnitudes.

This definition introduces no interaction law or
conservation assumption.
-/
def combined_magnitude_sq
    (A B : OscillatorState) : ℝ :=
  oscillator_magnitude_sq A +
    oscillator_magnitude_sq B

/--
The combined quadratic magnitude of two oscillator states
is nonnegative.

This follows from the nonnegativity of each state's
individual quadratic magnitude.
-/
theorem combined_magnitude_sq_nonneg
    (A B : OscillatorState) :
    0 ≤ combined_magnitude_sq A B := by
  unfold combined_magnitude_sq
  have hA := oscillator_magnitude_sq_nonneg A
  have hB := oscillator_magnitude_sq_nonneg B
  linarith

/--
The combined quadratic magnitude vanishes if and only if
both oscillator states are zero.

Because both individual quadratic magnitudes are
nonnegative, their sum can vanish only when each
individual magnitude vanishes.
-/
theorem combined_magnitude_sq_eq_zero_iff
    (A B : OscillatorState) :
    combined_magnitude_sq A B = 0 ↔
      (A.q = 0 ∧ A.p = 0) ∧
      (B.q = 0 ∧ B.p = 0) := by
  constructor
  · intro h
    have hA : oscillator_magnitude_sq A = 0 := by
      have hA_nonneg := oscillator_magnitude_sq_nonneg A
      have hB_nonneg := oscillator_magnitude_sq_nonneg B
      unfold combined_magnitude_sq at h
      linarith
    have hB : oscillator_magnitude_sq B = 0 := by
      have hA_nonneg := oscillator_magnitude_sq_nonneg A
      have hB_nonneg := oscillator_magnitude_sq_nonneg B
      unfold combined_magnitude_sq at h
      linarith
    exact ⟨(oscillator_magnitude_sq_eq_zero_iff A).mp hA,
      (oscillator_magnitude_sq_eq_zero_iff B).mp hB⟩
  · rintro ⟨hA, hB⟩
    unfold combined_magnitude_sq
    rw [
      (oscillator_magnitude_sq_eq_zero_iff A).mpr hA,
      (oscillator_magnitude_sq_eq_zero_iff B).mpr hB
    ]
    ring

/--
The combined quadratic magnitude is invariant under
exchange of the two oscillator states.

This is a consequence of commutativity of real addition
and does not depend on any interaction law.
-/
theorem combined_magnitude_sq_comm
    (A B : OscillatorState) :
    combined_magnitude_sq A B =
      combined_magnitude_sq B A := by
  unfold combined_magnitude_sq
  ring

/--
Each oscillator's quadratic magnitude is bounded above
by the combined quadratic magnitude.

This follows from nonnegativity of the other oscillator's
quadratic magnitude.
-/
theorem oscillator_magnitude_sq_le_combined
    (A B : OscillatorState) :
    oscillator_magnitude_sq A ≤
      combined_magnitude_sq A B := by
  unfold combined_magnitude_sq
  have hB := oscillator_magnitude_sq_nonneg B
  linarith

/-!
## Pairwise Interactions and Conservative Mixing

Introduce general pairwise oscillator interactions and
the predicate expressing conservation of combined
quadratic magnitude.

Normalized mixing provides a concrete family of
interactions whose conservation is established
independently of its definition.

The mixing coefficients act between oscillator states,
rather than between the internal coordinates of a
single state.
-/

/--
An arbitrary interaction between two oscillator states.

An interaction maps an ordered pair of oscillator states
to another ordered pair of oscillator states.

No linearity, continuity, conservation, reversibility,
or other restriction is imposed on the transformation.

In particular, the interaction need not preserve the
combined quadratic magnitude. Such properties are
specified separately by predicates on interactions.

The ordering of the pair identifies the two participants
and their corresponding outputs; it does not imply a
spatial direction or physical temporal ordering.
-/
def OscillatorInteraction :=
  OscillatorState × OscillatorState →
    OscillatorState × OscillatorState

/--
An interaction is conservative if it preserves the
combined quadratic magnitude of every input pair.

This property constrains the total magnitude, but does
not require the individual magnitudes to be preserved.
-/
def ConservesCombinedMagnitude
    (F : OscillatorInteraction) : Prop :=
  ∀ A B : OscillatorState,
    combined_magnitude_sq (F (A, B)).1 (F (A, B)).2 =
      combined_magnitude_sq A B

/--
Mix two oscillator states using normalized real coefficients.

The transformation is

    A' = c*A - s*B,
    B' = s*A + c*B,

where addition and scalar multiplication act componentwise
on the oscillator coordinates.

Unlike an internal oscillator rotation, this transformation
mixes the states of two distinct oscillators. The same
normalized coefficients act on their respective q and p
coordinate pairs.

This definition specifies a candidate interaction but does
not itself assert conservation.
-/
def mix_oscillators
    (r : NormalizedRotation) : OscillatorInteraction :=
  fun ⟨A, B⟩ =>
    (⟨r.c * A.q - r.s * B.q,
       r.c * A.p - r.s * B.p⟩,
     ⟨r.s * A.q + r.c * B.q,
       r.s * A.p + r.c * B.p⟩)

/--
Normalized linear mixing preserves the combined quadratic
magnitude of two oscillator states.

The interaction may change either individual magnitude,
but their sum remains unchanged.

The proof expands both output states. The cross terms
cancel, leaving the original combined magnitude multiplied
by `r.c² + r.s²`, which equals one by normalization.
-/
theorem mix_oscillators_conserves
    (r : NormalizedRotation) :
    ConservesCombinedMagnitude (mix_oscillators r) := by
  intro A B
  change
    (r.c * A.q - r.s * B.q) ^ 2 +
        (r.c * A.p - r.s * B.p) ^ 2 +
        ((r.s * A.q + r.c * B.q) ^ 2 +
          (r.s * A.p + r.c * B.p) ^ 2) =
      A.q ^ 2 + A.p ^ 2 + (B.q ^ 2 + B.p ^ 2)
  calc
    (r.c * A.q - r.s * B.q) ^ 2 +
        (r.c * A.p - r.s * B.p) ^ 2 +
        ((r.s * A.q + r.c * B.q) ^ 2 +
          (r.s * A.p + r.c * B.p) ^ 2) =
      (r.c ^ 2 + r.s ^ 2) *
        (A.q ^ 2 + A.p ^ 2 + (B.q ^ 2 + B.p ^ 2)) := by
          ring
    _ = A.q ^ 2 + A.p ^ 2 + (B.q ^ 2 + B.p ^ 2) := by
      rw [r.normalized]
      ring

/-!
## Inner Products and Magnitude Redistribution

Introduce the Euclidean inner product on the internal
two-component oscillator state space.

The inner product provides the cross terms appearing
in the individual quadratic magnitudes produced by
normalized mixing.

These identities describe how combined quadratic
magnitude can be redistributed between oscillator
states without changing its total value.

No physical spatial geometry or energy interpretation
is assumed.
-/

/--
The Euclidean inner product of two oscillator states.

For states `A = (q_A, p_A)` and `B = (q_B, p_B)`,
the inner product is

    q_A * q_B + p_A * p_B.

This is an algebraic relation between the coordinates
of two states. No physical interpretation is assumed.
-/
def oscillator_inner_product
    (A B : OscillatorState) : ℝ :=
  A.q * B.q + A.p * B.p

/--
The inner product of an oscillator state with itself
equals its quadratic magnitude.
-/
theorem oscillator_inner_product_self
    (A : OscillatorState) :
    oscillator_inner_product A A =
      oscillator_magnitude_sq A := by
  unfold oscillator_inner_product oscillator_magnitude_sq
  ring

/--
The quadratic magnitude of the subtractive output
c*A - s*B of a normalized mixing interaction.

The output magnitude consists of weighted contributions
from both input magnitudes and a cross term determined
by their inner product.

Although the interaction coefficients belong to a
`NormalizedRotation`, this identity follows from
polynomial expansion alone and does not use the
normalization condition.
-/
theorem mix_oscillators_subtractive_output_magnitude
    (r : NormalizedRotation)
    (A B : OscillatorState) :
    oscillator_magnitude_sq (mix_oscillators r (A, B)).1 =
      r.c ^ 2 * oscillator_magnitude_sq A +
      r.s ^ 2 * oscillator_magnitude_sq B -
      2 * r.c * r.s * oscillator_inner_product A B := by
  change
    (r.c * A.q - r.s * B.q) ^ 2 +
      (r.c * A.p - r.s * B.p) ^ 2 =
    r.c ^ 2 * (A.q ^ 2 + A.p ^ 2) +
      r.s ^ 2 * (B.q ^ 2 + B.p ^ 2) -
      2 * r.c * r.s * (A.q * B.q + A.p * B.p)
  ring

/--
The quadratic magnitude of the additive output
`s*A + c*B` of a normalized mixing interaction.

The cross term has the opposite algebraic sign from the
corresponding term in the subtractive-output formula.

Consequently, the cross terms cancel when the two
output magnitudes are added.
-/
theorem mix_oscillators_additive_output_magnitude
    (r : NormalizedRotation)
    (A B : OscillatorState) :
    oscillator_magnitude_sq (mix_oscillators r (A, B)).2 =
      r.s ^ 2 * oscillator_magnitude_sq A +
      r.c ^ 2 * oscillator_magnitude_sq B +
      2 * r.c * r.s * oscillator_inner_product A B := by
  change
    (r.s * A.q + r.c * B.q) ^ 2 +
      (r.s * A.p + r.c * B.p) ^ 2 =
    r.s ^ 2 * (A.q ^ 2 + A.p ^ 2) +
      r.c ^ 2 * (B.q ^ 2 + B.p ^ 2) +
      2 * r.c * r.s * (A.q * B.q + A.p * B.p)
  ring

/-!
## Magnitude Change and Interaction Properties

Quantify changes in combined quadratic magnitude under
general oscillator interactions.

Distinguish pointwise magnitude changes from properties
required to hold for every input pair, including
conservation, non-increase, and strict magnitude loss.

No physical dissipation or energy interpretation
is assumed.
-/

/--
The change in combined quadratic magnitude produced by
an interaction on a given pair of oscillator states.

The change is defined as the output combined magnitude
minus the input combined magnitude.

A negative value represents a decrease in quadratic
magnitude, zero represents no net change, and a positive
value represents an increase.

These are properties of the interaction on the specified
input pair. Conservation requires zero magnitude change
for every possible input pair.

No conservation assumption is imposed.
-/
def interaction_magnitude_change
    (F : OscillatorInteraction)
    (A B : OscillatorState) : ℝ :=
  combined_magnitude_sq (F (A, B)).1 (F (A, B)).2 -
    combined_magnitude_sq A B

/--
An interaction is non-increasing in combined quadratic
magnitude if its output magnitude never exceeds its
input magnitude.

Equality is permitted, so conservative interactions
satisfy this predicate as well.
-/
def NonIncreasingCombinedMagnitude
    (F : OscillatorInteraction) : Prop :=
  ∀ A B : OscillatorState,
    combined_magnitude_sq (F (A, B)).1 (F (A, B)).2 ≤
      combined_magnitude_sq A B

/--
An interaction exhibits strict loss if there exists an
input pair whose combined quadratic magnitude decreases.

This does not require every input to experience a loss,
nor does it exclude magnitude increases for other inputs.

Global non-increase is a separate property.
-/
def ExhibitsMagnitudeLoss
    (F : OscillatorInteraction) : Prop :=
  ∃ A B : OscillatorState,
    interaction_magnitude_change F A B < 0

/-!
## Inner-Product Identities and Bounds

Establish algebraic identities and inequalities relating
oscillator quadratic magnitudes and inner products.

The two-dimensional Lagrange identity provides a
sum-of-squares expression from which the squared
Cauchy–Schwarz inequality follows.

These results concern the internal oscillator state
space and require no assumptions about interactions
or physical spatial geometry.
-/

/--
The two-dimensional Lagrange identity for oscillator states.

The product of the individual quadratic magnitudes,
minus the square of their inner product, equals the
square of the determinant formed from their coordinates.

This is an exact polynomial identity and requires no
normalization or interaction assumptions.
-/
theorem oscillator_lagrange_identity
    (A B : OscillatorState) :
    oscillator_magnitude_sq A *
        oscillator_magnitude_sq B -
      oscillator_inner_product A B ^ 2 =
        (A.q * B.p - A.p * B.q) ^ 2 := by
  change
    (A.q ^ 2 + A.p ^ 2) *
        (B.q ^ 2 + B.p ^ 2) -
      (A.q * B.q + A.p * B.p) ^ 2 =
        (A.q * B.p - A.p * B.q) ^ 2
  ring

/--
The Cauchy–Schwarz inequality for oscillator states.

The square of the inner product is bounded above by
the product of the individual quadratic magnitudes.

The proof follows from the two-dimensional Lagrange
identity and the nonnegativity of a real square.
-/
theorem oscillator_inner_product_sq_le
    (A B : OscillatorState) :
    oscillator_inner_product A B ^ 2 ≤
      oscillator_magnitude_sq A *
        oscillator_magnitude_sq B := by
  have h := oscillator_lagrange_identity A B
  have hs : 0 ≤ (A.q * B.p - A.p * B.q) ^ 2 :=
    sq_nonneg _
  linarith

/--
An interaction conserves combined quadratic magnitude
if and only if its net magnitude change vanishes for
every input pair.

This connects the conservation predicate with the
general definition of magnitude change.
-/
theorem conserves_iff_zero_magnitude_change
    (F : OscillatorInteraction) :
    ConservesCombinedMagnitude F ↔
      ∀ A B : OscillatorState,
        interaction_magnitude_change F A B = 0 := by
  constructor
  · intro h A B
    unfold interaction_magnitude_change
    rw [h A B]
    ring
  · intro h A B
    have hz := h A B
    unfold interaction_magnitude_change at hz
    linarith

/--
Every conservative interaction is non-increasing in
combined quadratic magnitude.

A conservative interaction satisfies the non-increasing
condition with equality for every input pair.
-/
theorem conserves_implies_nonincreasing
    (F : OscillatorInteraction)
    (h : ConservesCombinedMagnitude F) :
    NonIncreasingCombinedMagnitude F := by
  intro A B
  rw [h A B]

/--
Normalized mixing produces zero net change in combined
quadratic magnitude for every input pair.

This follows from the conservation theorem for mixing
and the general equivalence between conservation and
zero magnitude change.
-/
theorem mix_oscillators_zero_magnitude_change
    (r : NormalizedRotation)
    (A B : OscillatorState) :
    interaction_magnitude_change (mix_oscillators r) A B = 0 := by
  exact
    (conserves_iff_zero_magnitude_change
      (mix_oscillators r)).mp
        (mix_oscillators_conserves r) A B

/-!
## Uniform Scaling and Magnitude Loss

Define uniform scaling interactions and establish how
they transform combined quadratic magnitude.

The scaling coefficient determines whether magnitude
is conserved, reduced, or increased.

In particular, scaling by one half provides a concrete
interaction that is globally non-increasing and exhibits
strict magnitude loss.

These results concern quadratic magnitude only; no
physical dissipation mechanism is assumed.
-/

/--
Uniformly scale both oscillator states by a real coefficient.

The same coefficient is applied to all four coordinates.

This defines a candidate interaction without imposing
conservation, non-increase, or any other restriction
on the scaling coefficient.
-/
def scale_oscillators
    (k : ℝ) : OscillatorInteraction :=
  fun ⟨A, B⟩ =>
    (⟨k * A.q, k * A.p⟩,
     ⟨k * B.q, k * B.p⟩)

/--
Uniform scaling multiplies the combined quadratic
magnitude by the square of the scaling coefficient.

This identity holds for every real scaling coefficient,
without any conservation or non-increase assumption.
-/
theorem scale_oscillators_magnitude
    (k : ℝ) (A B : OscillatorState) :
    combined_magnitude_sq
        (scale_oscillators k (A, B)).1
        (scale_oscillators k (A, B)).2 =
      k ^ 2 * combined_magnitude_sq A B := by
  change
    (k * A.q) ^ 2 + (k * A.p) ^ 2 +
      ((k * B.q) ^ 2 + (k * B.p) ^ 2) =
    k ^ 2 * (A.q ^ 2 + A.p ^ 2 +
      (B.q ^ 2 + B.p ^ 2))
  ring

/--
Uniform scaling is non-increasing in combined quadratic
magnitude whenever the square of the scaling coefficient
is at most one.

The condition allows conservation when `k² = 1`
and loss when `k² < 1` and the input magnitude is positive.
-/
theorem scale_oscillators_nonincreasing
    (k : ℝ) (hk : k ^ 2 ≤ 1) :
    NonIncreasingCombinedMagnitude (scale_oscillators k) := by
  intro A B
  rw [scale_oscillators_magnitude]
  have hQ : 0 ≤ combined_magnitude_sq A B :=
    combined_magnitude_sq_nonneg A B
  calc
    k ^ 2 * combined_magnitude_sq A B
        ≤ 1 * combined_magnitude_sq A B :=
      mul_le_mul_of_nonneg_right hk hQ
    _ = combined_magnitude_sq A B := one_mul _

/--
Scaling by one half exhibits strict loss of combined
quadratic magnitude.

The witness consists of one unit-magnitude oscillator
and one zero oscillator. The combined magnitude decreases
from one to one quarter.

Together with `scale_oscillators_nonincreasing`, this
demonstrates that a non-increasing interaction need
not be conservative.
-/
theorem scale_half_exhibits_loss :
    ExhibitsMagnitudeLoss (scale_oscillators (1 / 2)) := by
  refine ⟨⟨1, 0⟩, ⟨0, 0⟩, ?_⟩
  unfold interaction_magnitude_change
  norm_num [scale_oscillators, combined_magnitude_sq,
    oscillator_magnitude_sq]

/--
The net change in combined quadratic magnitude under
uniform scaling.

The change is the original combined magnitude multiplied
by the difference between the squared scaling coefficient
and one.

This identity holds for every real scaling coefficient
and every pair of oscillator states.
-/
theorem scale_oscillators_magnitude_change
    (k : ℝ) (A B : OscillatorState) :
    interaction_magnitude_change (scale_oscillators k) A B =
      (k ^ 2 - 1) * combined_magnitude_sq A B := by
  unfold interaction_magnitude_change
  rw [scale_oscillators_magnitude]
  ring

/-!
## Composition of Oscillator Interactions

Define sequential composition of arbitrary oscillator
interactions and establish its algebraic properties.

Conservation and non-increase of combined quadratic
magnitude are preserved when interactions satisfying
the corresponding properties are composed.

Composition is associative and admits the identity
interaction as a two-sided identity.

These results establish the foundations for the monoid
structure introduced subsequently.
-/

/--
Compose two oscillator interactions.

`compose_interactions F G` applies `G` first,
followed by `F`.

No conservation or magnitude restriction is imposed
on either interaction.
-/
def compose_interactions
    (F G : OscillatorInteraction) : OscillatorInteraction :=
  fun pair => F (G pair)

/--
The composition of two conservative oscillator
interactions is conservative.

The interaction `G` is applied first, followed by `F`.
Conservation at each stage implies conservation of
the combined quadratic magnitude across the composition.
-/
theorem compose_interactions_conserves
    (F G : OscillatorInteraction)
    (hF : ConservesCombinedMagnitude F)
    (hG : ConservesCombinedMagnitude G) :
    ConservesCombinedMagnitude (compose_interactions F G) := by
  intro A B
  cases hpair : G (A, B) with
  | mk C D =>
      unfold compose_interactions
      rw [hpair]
      calc
        combined_magnitude_sq (F (C, D)).1 (F (C, D)).2 =
            combined_magnitude_sq C D := hF C D
        _ = combined_magnitude_sq A B := by
          simpa only [hpair] using hG A B

/--
The composition of two non-increasing oscillator
interactions is non-increasing.

Since neither interaction increases the combined
quadratic magnitude, their successive application
also cannot increase it.
-/
theorem compose_interactions_nonincreasing
    (F G : OscillatorInteraction)
    (hF : NonIncreasingCombinedMagnitude F)
    (hG : NonIncreasingCombinedMagnitude G) :
    NonIncreasingCombinedMagnitude
      (compose_interactions F G) := by
  intro A B
  cases hpair : G (A, B) with
  | mk C D =>
      unfold compose_interactions
      rw [hpair]
      calc
        combined_magnitude_sq (F (C, D)).1 (F (C, D)).2 ≤
            combined_magnitude_sq C D := hF C D
        _ ≤ combined_magnitude_sq A B := by
          simpa only [hpair] using hG A B

/--
Composing a conservative interaction after a non-increasing
interaction produces a non-increasing interaction.

The conservative stage preserves the magnitude remaining
after the non-increasing stage.
-/
theorem compose_conserving_after_nonincreasing
    (F G : OscillatorInteraction)
    (hF : ConservesCombinedMagnitude F)
    (hG : NonIncreasingCombinedMagnitude G) :
    NonIncreasingCombinedMagnitude (compose_interactions F G) := by
  exact compose_interactions_nonincreasing F G
    (conserves_implies_nonincreasing F hF) hG

/--
Composing a non-increasing interaction after a conservative
interaction produces a non-increasing interaction.

The conservative stage preserves the initial magnitude,
and the subsequent interaction cannot increase it.
-/
theorem compose_nonincreasing_after_conserving
    (F G : OscillatorInteraction)
    (hF : NonIncreasingCombinedMagnitude F)
    (hG : ConservesCombinedMagnitude G) :
    NonIncreasingCombinedMagnitude (compose_interactions F G) := by
  exact compose_interactions_nonincreasing F G
    hF (conserves_implies_nonincreasing G hG)

/--
Composition of oscillator interactions is associative.

For any three interactions `F`, `G`, and `H`,
the grouping of successive applications does not
affect the resulting transformation.

The interaction `H` is applied first, followed by
`G`, and finally `F`, regardless of grouping.

This property follows from associativity of function
composition and requires no conservation assumptions.
-/
theorem compose_interactions_assoc
    (F G H : OscillatorInteraction) :
    compose_interactions (compose_interactions F G) H =
      compose_interactions F (compose_interactions G H) := by
  rfl

/--
The identity oscillator interaction.

This interaction returns its input pair unchanged
and therefore preserves the combined quadratic magnitude.
-/
def identity_interaction : OscillatorInteraction :=
  fun pair => pair

/--
The identity interaction is a left identity under
interaction composition.

Applying an interaction `F` and then the identity
interaction gives the same result as applying `F` alone.
-/
theorem compose_interactions_identity_left
    (F : OscillatorInteraction) :
    compose_interactions identity_interaction F = F := by
  rfl

/--
The identity interaction is a right identity under
interaction composition.

Applying the identity interaction before an
interaction `F` gives the same result as applying `F`
alone.
-/
theorem compose_interactions_identity_right
    (F : OscillatorInteraction) :
    compose_interactions F identity_interaction = F := by
  rfl


/-!
## Monoid Structure of Oscillator Interactions

Oscillator interactions form a monoid under composition.

The multiplication operation applies the right-hand
interaction first, followed by the left-hand interaction.

Composition is associative, and `identity_interaction`
is a two-sided identity.

No conservation, non-increase, or invertibility
assumption is imposed on general interactions.
-/

/--
Equip oscillator interactions with a monoid structure
under function composition.

Multiplication is `compose_interactions`, and the
identity element is `identity_interaction`.

The required algebraic laws follow from the previously
established associativity and identity theorems.

The monoid structure requires neither commutativity
nor invertibility. In particular, arbitrary oscillator
interactions need not possess inverses.
-/
instance : Monoid OscillatorInteraction where
  mul := compose_interactions
  one := identity_interaction
  mul_assoc := compose_interactions_assoc
  one_mul := compose_interactions_identity_left
  mul_one := compose_interactions_identity_right

/--
Monoid multiplication of oscillator interactions is
their previously defined composition operation.

This equality holds definitionally because the
`Monoid` instance uses `compose_interactions`
as its multiplication.
-/
theorem oscillator_interaction_mul_def
    (F G : OscillatorInteraction) :
    F * G = compose_interactions F G := by
  rfl

/--
The identity interaction preserves the combined
quadratic magnitude of every pair of oscillator states.
-/
theorem identity_interaction_conserves :
    ConservesCombinedMagnitude identity_interaction := by
  intro A B
  rfl

/--
The identity interaction is non-increasing in combined
quadratic magnitude.

It satisfies the inequality with equality.
-/
theorem identity_interaction_nonincreasing :
    NonIncreasingCombinedMagnitude identity_interaction := by
  exact conserves_implies_nonincreasing
    identity_interaction identity_interaction_conserves


/-!
## Submonoids of Oscillator Interactions

Conservative and non-increasing interactions each form
a submonoid of the monoid of all oscillator interactions.

Both classes contain the identity interaction and are
closed under composition.

Every conservative interaction is non-increasing, so
the conservative submonoid is contained in the
non-increasing submonoid.
-/

/--
The submonoid of interactions preserving the combined
quadratic magnitude of every input pair.

Membership is characterized by
`ConservesCombinedMagnitude`.
-/
def conservative_interactions : Submonoid OscillatorInteraction where
  carrier := {F | ConservesCombinedMagnitude F}
  one_mem' := identity_interaction_conserves
  mul_mem' := by
    intro F G hF hG
    rw [oscillator_interaction_mul_def]
    exact compose_interactions_conserves F G hF hG

/--
The submonoid of interactions that never increase
the combined quadratic magnitude.

Membership is characterized by
`NonIncreasingCombinedMagnitude`.

Conservative interactions belong to this submonoid,
but strict loss is permitted.
-/
def nonincreasing_interactions : Submonoid OscillatorInteraction where
  carrier := {F | NonIncreasingCombinedMagnitude F}
  one_mem' := identity_interaction_nonincreasing
  mul_mem' := by
    intro F G hF hG
    rw [oscillator_interaction_mul_def]
    exact compose_interactions_nonincreasing F G hF hG

/--
Every conservative oscillator interaction belongs to
the submonoid of non-increasing interactions.

This follows from the general implication that
conservation entails non-increase.
-/
theorem conservative_interactions_le_nonincreasing :
    conservative_interactions ≤ nonincreasing_interactions := by
  intro F hF
  exact conserves_implies_nonincreasing F hF

/--
The conservative interactions form a proper submonoid
of the non-increasing interactions.

Uniform scaling by one half is non-increasing but
exhibits strict loss for some input pair, so it
cannot conserve combined magnitude universally.

This establishes that the non-increasing submonoid
contains interactions that are not conservative.
-/
theorem conservative_interactions_lt_nonincreasing :
    conservative_interactions < nonincreasing_interactions := by
  apply lt_of_le_of_ne
    conservative_interactions_le_nonincreasing
  intro h
  have hmem :
      scale_oscillators (1 / 2 : ℝ) ∈
        nonincreasing_interactions := by
    exact scale_oscillators_nonincreasing (1 / 2) (by norm_num)
  have hcons :
      ConservesCombinedMagnitude
        (scale_oscillators (1 / 2)) := by
    rw [← h] at hmem
    exact hmem
  obtain ⟨A, B, hloss⟩ := scale_half_exhibits_loss
  have hzero :=
    (conserves_iff_zero_magnitude_change
      (scale_oscillators (1 / 2))).mp hcons A B
  linarith

end StandingWave
