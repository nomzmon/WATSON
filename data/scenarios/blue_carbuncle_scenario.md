# Scenario: The Blue Carbuncle

Based on "The Adventure of the Blue Carbuncle", a short story from The Adventures of Sherlock Holmes by Arthur Conan Doyle.

## Scenario setup

Set the story in Victorian London, two days after Christmas. The user takes Dr. Watson's place as Sherlock Holmes's trusted companion. Together, they investigate how a stolen jewel ended up inside a Christmas goose.

Begin at 221B Baker Street. Peterson, a commissionaire, has recovered a hat and a goose after their owner fled a street disturbance. He brought them to Holmes, who allowed him to take the goose home to eat. Peterson has now returned with a blue gemstone that his wife found inside the bird. The hat and stone are available to examine, and Peterson can answer questions about what happened.

## How to run the scenario

Holmes begins with only the knowledge and traits in the persona profile created beforehand. He learns the case alongside the user. Reveal clues through the user's investigation, then let Holmes use the profile and the evidence revealed so far to interpret them, offer hints, and suggest leads.

Use the following events as a guide. Allow natural questions and reasonable variations in order. If the user gets stuck, Holmes can explain what the evidence suggests and point them toward a useful action.

## Opening scene

Peterson places a blue stone on the table beside the battered hat. "My wife found it inside the goose, Mr. Holmes." Holmes turns toward you. "An unexpected discovery, my dear friend. Let us examine what Peterson has brought us."

## Investigation stages and clues

Reveal clues as the user investigates. Holmes learns them at the same time and may offer the deductions below once the supporting evidence is available. These are possible interpretations, not information he knows in advance.

| Stage | Location | Clues and how to find them |
|---|---|---|
| Identify the jewel and owner | Baker Street | Examine the stone and check the newspaper. The report's description connects it to the Countess of Morcar's stolen jewel. It names John Horner as the accused and James Ryder as a witness. Inspect the hat for H. B. Ask Peterson about the label, which read "For Mrs. Henry Baker." |
| Question Henry Baker | Baker Street | Ask about the original goose or offer its remains and a replacement. Baker accepts the replacement and has no interest in the remains. Ask where he got the bird. He identifies the Alpha Inn's goose club. |
| Trace the supplier | Alpha Inn | Ask the landlord where the goose club's birds came from. He names Breckinridge in Covent Garden. |
| Examine market records | Covent Garden | Ask Breckinridge about his supplier and examine his records. They identify Mrs. Oakshott of Brixton Road. His complaints reveal repeated inquiries about the birds. Investigating those inquiries brings the pair into contact with an anxious man seeking a particular goose. |
| Identify the stranger | Covent Garden | Question the man about his identity and interest in the goose. He first calls himself John Robinson, then admits he is James Ryder. |
| Establish the solution | Baker Street | Confront Ryder with the jewel and the evidence connecting him to the bird. He confesses to stealing it with Catherine Cusack's involvement and blaming Horner. He hid the jewel in a goose at his sister Mrs. Oakshott's home, then took the wrong bird after confusing two similar geese. |

## Ending

To follow the original story's ending, Holmes releases Ryder and expects the case against Horner to collapse when Ryder fails to appear as a witness. The user can discuss or challenge this decision with Holmes before the scenario ends.

## Reminders

Keep this guide's unrevealed clues and solution with the person running the scenario. Holmes receives the established persona profile, the opening situation, and each clue as it is discovered. He learns names, connections, and the culprit through the investigation. Base his deductions on the evidence he and the user have encountered.

The investigation above follows the story's main route. Its opening position and interactive exchanges are adaptations. Dialogue can be written naturally around these events, with the user occupying Watson's companion role.

## How WATSON uses it

- Holmes is given only the opening situation, as `case_context` in the dialogue files (`scripted_dialogues/`, `long_dialogues/`). The clues reach him through the visitor's messages.
- The Dialogue Flow Manager's five stages map onto the stages above: introduction (opening scene), evidence (identify the jewel and owner, question Henry Baker), suspects (trace the supplier, examine market records), deduction (identify the stranger), resolution (establish the solution and the ending).