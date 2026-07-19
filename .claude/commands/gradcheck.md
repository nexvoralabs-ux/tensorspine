Run: pytest tests/test_gradcheck.py -v

For each failure:
1. Report which operation and which composite expression failed
2. Report analytical vs numerical value and the relative error
3. Classify the failure using the debugging protocol in CLAUDE.md
4. State the CATEGORY of bug, not the fix
5. Ask the user to state the derivative rule they implemented

If the error is ~2x expected, say: "check accumulation."
If gradients are exactly zero, say: "check the seed."
If shallow nodes are right and deep ones wrong, say: "check topo order."

Do not open engine.py and fix it.
