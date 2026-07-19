Administer the closed-book assessment. Tell the user to close their
editor and use paper only. 45 minutes, 8 questions, one at a time.

1. Derive d/dx tanh(x) from the exponential definition.
2. Draw the computation graph for L = (w·x + b − y)². Label every
   intermediate node. Compute ∂L/∂w by hand.
3. Node `a` feeds both `b` and `c`, which both feed `L`. Write ∂L/∂a
   and explain why it is a sum.
4. In three sentences: why does reverse-mode beat forward-mode for
   training neural networks?
5. A network won't learn; all gradients are zero after step one. Give
   three possible causes.
6. Why does backward require reverse topological order? What
   specifically breaks in arbitrary order?
7. Given only +, * and pow, express division and subtraction. Argue
   their gradients are correct without deriving them.
8. Explain mechanically why 50 stacked tanh layers train poorly.

Grade each 0/0.5/1. Pass is 6/8.

If they pass: confirm week 1 complete, list what this unlocks.
If they fail: name the specific objectives that failed and say plainly
that moving to tensors now would build on sand.

Do not accept vague answers. Push for mechanism.
