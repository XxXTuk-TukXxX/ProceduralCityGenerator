import random
from typing import List, Set

class UniquePicker:
    """
    Draws up to **three** different numbers from {0,1,2,3}.

    Per-range probabilities
      • 1st pick: 80 % chance
      • 2nd pick: 20 % chance
      • 3rd pick:  1 % chance
    Once a number is used it never re-appears *and* the global total
    can never exceed three.
    """

    _POOL: Set[int] = {0, 1, 2, 3}
    _MAX_PICKS = 3                   # ← global ceiling

    def __init__(self) -> None:
        self._remaining: Set[int] = set(self._POOL)
        self._picked_so_far = 0

    # ───────────────────────────────────
    def pick_range(self) -> List[int]:
        """Return 0-to-3 unique numbers for one call (‘range’)."""
        group: List[int] = []

        if self._try_pick(group, 0.80):          # 1st (80 %)
            if self._try_pick(group, 0.20):      # 2nd (20 %)
                self._try_pick(group, 0.01)      # 3rd  (1 %)

        return group

    # ───────────────────────────────────
    def _try_pick(self, bucket: List[int], prob: float) -> bool:
        """
        With probability *prob*, move one unused number into *bucket*.
        Returns True if a number was added.
        """
        if (self._picked_so_far >= self._MAX_PICKS          # global cap reached
                or not self._remaining                      # nothing left
                or random.random() > prob):                 # probability says “skip”
            return False

        choice = random.choice(tuple(self._remaining))
        self._remaining.remove(choice)

        bucket.append(choice)
        self._picked_so_far += 1
        return True

    # ───────────────────────────────────
    def remaining(self) -> Set[int]:
        """Numbers that could still be drawn (may be empty)."""
        return set(self._remaining)
    



# picker = UniquePicker()

# print(picker.pick_range())   # e.g. [3, 0]
# print(picker.pick_range())   # e.g. [1]   (can never choose two now)
# print("unused numbers →", picker.remaining())   # {2}
