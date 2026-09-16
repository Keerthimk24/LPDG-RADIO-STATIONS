# AI Usage Disclosure

## Tools Used

| Tool | What I used it for |
|---|---|
| **Claude (Anthropic)** | Architecture design, code generation for boilerplate modules, test fixture creation, documentation drafting |
| **GitHub Copilot** | Inline code completion while writing feature engineering functions |

## How I Used AI

- **Architecture planning:** I described the challenge requirements and asked for help designing the module structure, Docker setup, and CI/CD pipeline. The AI suggested the separation of concerns between `src/data/`, `src/features/`, `src/model/`, and `src/monitoring/`.
- **Boilerplate code:** Data loading functions, test fixtures with synthetic data, Dockerfile and docker-compose.yml templates.
- **Documentation:** First drafts of DECISIONS.md and this file. I revised them to match my actual reasoning.
- **Debugging:** When gateway IDs were not joining correctly across datasets (colon vs bare hex format), AI helped identify the normalisation issue.

## One Thing AI Got Wrong That I Spotted

**The cost function breakeven calculation was initially wrong.**

The AI first computed the breakeven probability as:

```
P(broken) > 380/600 = 0.633
```

This is incorrect. The correct formulation for the visit value is:

```
Visit_value = P(broken) × €600 - (1 - P(broken)) × €380
```

Setting to zero: `P × 600 = (1-P) × 380` → `P = 380/980 ≈ 0.388`

The AI's version only considered the cost of a false negative (€600) in the denominator, ignoring that a false positive also has a cost (€380) that applies to the complement `(1-P)`. This would have set the threshold too high, meaning we would visit too few gateways and miss broken ones — exactly the wrong direction given that false negatives cost more.

I caught this during the cost function unit tests (`test_breakeven_probability`) where the assertion verified the breakeven point mathematically. The test would have failed with the AI's original formula.

This matters because the entire ranking system depends on this calculation. Getting the breakeven wrong would shift all 15 visit selections.
