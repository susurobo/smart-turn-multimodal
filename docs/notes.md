# intro

Physical AI is rarely see together in one sentence with voice AI. 

# limitationa
Dataset variety -- right now only one, mostly unscripted monologues between interviewer prompts. Will not generalize well on general conversations.

Trained on X unique videos with the minority class label (complete) duplicated for class balance and then scaled x times to make 10% of Audio training data.

still triggered only by the VAD-detected silence. In reality people often know when the turns ends before the silence happens.