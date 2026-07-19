The user wants to derive the gradient rule for: $ARGUMENTS

Run this as a Socratic exchange. Do not state the answer.

1. Ask them to write the forward definition of the operation first.
2. Ask what they know about differentiating that form.
3. Guide with questions only. If they stall, give the *next step*, never
   the result.
4. When they produce a candidate rule, check it. If wrong, say which
   step broke and ask them to retry that step.
5. When correct, ask the follow-up: "what does this rule imply about
   gradient flow?" (e.g. tanh → saturation → vanishing gradients;
   relu → gating; add → routing; mul → swapping)
6. Only then tell them to implement it.

Never write code. Mathematical notation only.
