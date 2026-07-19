# Derivations

> Fill this in **by hand**, before (or while) you implement the matching stub.
> The rule of the project: if you can't state the derivative in your own words
> here, you're not ready to write the code for it. Work each section out
> yourself — nothing below is answered for you.

---

## 1. The chain rule for shared nodes

A node `a` feeds into several downstream nodes `u₁, u₂, …, uₖ`, all of which
eventually reach the final scalar loss `L`.

- Write `dL/da` in terms of the `dL/duᵢ` and the local `duᵢ/da`.
- Why is it a **sum** over the consumers rather than any single one?
- Connect this to the code rule "accumulate with `+=`, never `=`". What exactly
  would `=` compute instead, and why is it wrong?

_your derivation:_


---

## 2. `d/dx tanh(x)` from the exponential definition

Start from `tanh(x) = (eˣ − e⁻ˣ) / (eˣ + e⁻ˣ)`.

- Differentiate using the quotient rule.
- Simplify until you can express the result purely in terms of `tanh(x)`.
- What is the maximum value of this derivative, and at what `x`? (This matters
  for experiment 05.)

_your derivation:_


---

## 3. Local gradients of each primitive

For each op, state the output and its partial derivative(s) w.r.t. each input.

### `+` :  `c = a + b`
- `∂c/∂a = ?`
- `∂c/∂b = ?`

### `*` :  `c = a · b`
- `∂c/∂a = ?`
- `∂c/∂b = ?`

### `pow` :  `c = aᵏ` (k constant)
- `∂c/∂a = ?`

### `exp` :  `c = eᵃ`
- `∂c/∂a = ?`

### `tanh` :  `c = tanh(a)`
- `∂c/∂a = ?`

### `relu` :  `c = max(0, a)`
- `∂c/∂a = ?`  (and what convention do you take at `a = 0`?)

### derived ops — express each as a composition, then say why no new local rule is needed
- `−a` = ?
- `a − b` = ?
- `a / b` = ?


---

## 4. Why reverse-mode beats forward-mode for a scalar output

Consider a function `f : ℝⁿ → ℝ` (many inputs, one scalar loss).

- How many forward-mode passes would it take to get all `n` partial derivatives?
- How many reverse-mode passes?
- Where does the cost asymmetry come from? Frame it in terms of propagating a
  vector vs a scalar through the graph.

_your reasoning:_


---

## 5. Why `backward` requires reverse topological order

- What must be true about a node's `grad` *before* its `_backward` closure runs?
- Show that visiting nodes in reverse topological order guarantees it.
- Give a small graph where visiting in the wrong order produces a wrong gradient.

_your reasoning:_


---

## 6. MSE gradient w.r.t. the prediction

For a single sample, `L = (ŷ − y)²` with `y` a constant target.

- `∂L/∂ŷ = ?`
- For a mean over `N` samples, `L = (1/N) Σ (ŷᵢ − yᵢ)²`, what is `∂L/∂ŷⱼ`?
- Sanity check: does this match what `backward()` deposits on a prediction node
  in your engine? Verify against one of the tests.

_your derivation:_
