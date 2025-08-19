---
title: "Stay or Switch, A Frequentist Approach to Monte Hall"
date: "2023-06-29"
---

The Monte Hall is a famous brain teaser in the form of a probability puzzle.
For the uninitiated, the problem is as follows: Consider three doors where
two of the doors have nothing behind them and one of the doors has a prize.
You are instructed to select the door with the prize.

The game show host opens a door that is different than the door you chose
that does not contain the prize. You now know that the prize is either behind
the door you originally chose or the door that the game show host did not open.
Should you stay at your current door, or switch to the unopened door?

Initially, most people would say that it shouldn’t matter. After all, you chose
the first door randomly. As it turns out, you have a 1-in-3 chance of being correct
if you stay and a 2-in-3 chance of being correct if you switch! This seems 
impossible. The frequentist in me knows that we can simulate this using a scripting
language!

Our simulation will randomly select two integers between 1 and 3; the first
integer represents the contestant’s first choice whereas the second integer
represents the correct answer (the door with the prize). We also denote the simulation
ID in the first column.

Next we randomly determine which door will be revealed to the contestant. If the
correct answer and the contestant’s choice are different, then the door that is
revealed is whatever door is left (i.e., if the answer is door 1 and the choice is
door 2 then door 3 is revealed).

If the correct answer and the contestant’s choice are the same, then the revealed
door is chosen at random between the remaining doors (i.e., if the contestant chose
door 1 and the answer is door 1, then either door 2 or 3 is revealed).

Finally, we need to determine which door is the switch door. That is, whatever door
has not already been chosen and not revealed. If door 1 was originally chosen and
door 2 was revealed, then door 3 is the switch door (regardless of the correct answer)

| row_id | answer | choice | reveal | switch |
| - | - | - | - | - |
| 1 | 1 | 3 | 2 | 1 | 
| 2 | 2 | 2 | 1 | 3 | 

Now we can use R and the tidyverse to construct a function which will simulate
Monte Hall’s problem many times. Because vectorized operations in R are fast,
we will stay within that constraint for our function. Additionally, this solution
leverages the row ID to prevent grouping the data frame into n groups providing 
further performance benefits!

```r
simulate <- function(n = 1e6) {
  # Simulation of the Monte Hall problem
  
  values <- seq(3)
  reveal <- values
  switch <- values
  
  tibble::tibble(
    row_id = seq(n),
    answer = sample(values, n, replace = TRUE),
    choice = sample(values, n, replace = TRUE)
  ) |>
    tidyr::expand_grid(reveal) |>
    dplyr::filter(
      reveal != answer,
      reveal != choice
    ) |>
    dplyr::distinct(
      row_id,
      answer,
      choice,
      .keep_all = TRUE
    ) |>
    tidyr::expand_grid(switch) |>
    dplyr::filter(
      switch != choice,
      switch != reveal
    ) |>
    dplyr::summarize(
      stay = mean(choice == answer),
      swap = mean(switch == answer)
    )
}
```

This function can carry out over 1.5 million simulations in around
1 second on my machine thanks to the power of vectorization! The function
starts by randomly selecting the answer and choice.

It then finds out which door to reveal by looking for the door not already
chosen by the answer or choice. If the answer and choice are the same,
then distinct ensures there is only one revealed door for each combination.

Finally, we determine which is the switch door. Because choice and reveal
are never the same, there is no need to ensure distinct records here. The
resulting data is summarized to determine the number of times the contestant
would be correct if they stayed versus swapped to a the other door.

Over the 1 million simulations I performed, staying at the same door was
correct 334,160 times whereas switching was correct 665,840 times. This is
almost exactly 1-in-3 and 2-in-3 for staying and switching respectively
sufficiently satisfying the frequentist approach to the Monte Hall problem!
