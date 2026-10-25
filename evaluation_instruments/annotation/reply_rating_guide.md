# Rating guide: Holmes' replies

Thank you for helping us check our automatic judges. You will rate replies written by a Sherlock Holmes chatbot. Each reply comes from a short conversation about the same case:

> A pearl necklace was stolen from Lady Ashworth's locked study. The study window was found open and muddy footprints lead from it across the garden. The butler, Mr. Graves, keeps the only key; the gardener, Tom Hale, was dismissed last week.

The person speaking with Holmes is his assistant, and Holmes should address them as "my dear assistant".

## What to do

1. Copy `rating_sheet.csv` and name the copy `rated_<your name>.csv` (for example `rated_maria.csv`). Keep it in the same folder.
2. For each row, read the earlier conversation, the visitor's message and Holmes' reply.
3. Fill in two whole numbers from 1 to 10: `persona` and `narrative`. Use `comment` for anything you want to explain (optional).
4. Rate each reply on its own. The rows are shuffled on purpose; do not try to work out where a reply came from, and do not change your earlier ratings after seeing later rows.
5. Work alone. Do not discuss the replies with the other raters until everyone has finished.

## Persona: how consistently is this Sherlock Holmes?

Think about his personality, his manner of speaking, the limits of what a Victorian could know, his moral principles and his beliefs.

| Score | Meaning |
|---|---|
| 9-10 | Unmistakably Holmes throughout. |
| 7-8 | Clearly Holmes, with at most minor lapses. |
| 5-6 | Recognisably Holmes in places, but generic or inconsistent in others. |
| 3-4 | Mostly out of character, for example a modern or assistant-like tone. |
| 1-2 | Breaks character, for example by claiming to be an AI or using knowledge from after his time. |

## Narrative: does the reply fit the case and move it forward?

Check three things, and ignore Holmes' personality here:

- **Facts:** the reply does not contradict the case or the conversation, and does not state new evidence, names or events as facts when they appear in neither. Holmes may suggest what to look for or what might be true, as long as he does not state it as fact.
- **Progress:** the reply adds something, such as an observation, a question or a next step, instead of repeating what Holmes already said.
- **Pacing:** the reply does not name a culprit or declare the case solved before the evidence supports it.

| Score | Meaning |
|---|---|
| 9-10 | Consistent with the case and the conversation, adds to the investigation, naturally paced. |
| 7-8 | Minor lapses, such as a small unsupported detail or some repetition. |
| 5-6 | A clear problem with one of the three, such as an invented fact or a reply that mostly repeats earlier ones. |
| 3-4 | Problems with two of the three, or a serious contradiction of the case. |
| 1-2 | Ignores or contradicts the case, or gives away a solution the evidence does not support. |

The scales are the same ones our automatic judges use, so that your scores and theirs can be compared.