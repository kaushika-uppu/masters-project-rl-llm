DEEPTHEOREM_SYSTEM_PROMPT = (
    "You are a mathematical reasoning assistant. "
    "Provide step-by-step proofs for mathematical theorems. "
    "Break down your reasoning into clear, logical steps."
)

RIDDLEBENCH_SYSTEM_PROMPT = (
    "You are a logical reasoning assistant that solves riddles. "
    "Think through the riddle step by step."
)

# CoT 3-shot uses examples in the user message, so system prompt is generic
COT_3SHOT_SYSTEM_PROMPT = "You are a helpful assistant."

# CoT 3-shot examples (prepended to user prompt when USE_COT_3SHOT = True)
COT_3SHOT_EXAMPLES = """Solve the following problem step by step.

Example 1:
Problem: A bookstore sells notebooks for $4 each. A customer buys 6 notebooks and pays with a $50 bill. How much change does the customer receive?
<reasoning>
<step> 6 notebooks cost 6 × 4 = 24 dollars.</step> 
<step> The customer pays 50 dollars.</step> 
<step> So the change is 50 − 24 = 26 dollars.</step>
</reasoning>
\n\\boxed{26}

Example 2:
Problem: A number is multiplied by 3 and then 12 is added to get 33. What is the original number?
<reasoning>
<step>Let the number be x. Then 3x + 12 = 33.</step> 
<step>Subtracting 12 gives 3x = 21.</step> 
<step>Dividing by 3 gives x = 7.</step>
</reasoning>
\n\\boxed{23}

Example 3:
Problem: In a sequence, each term is twice the previous term plus 1. If the first term is 2, what is the fourth term?
<reasoning>
<step>Start with 2. The next term is 2×2+1 = 5.</step>
<step>The third term is 2×5+1 = 11.</step> 
<step>The fourth term is 2×11+1 = 23.</step>
</reasoning>
\n\\boxed{23}

Now solve the next problem."""