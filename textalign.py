

import numpy as np

from numpy import unravel_index


class SmithWaterman:

    def __init__(self, s1, s2, match_score=10, mismatch_score=-5,
            gap_penalty=10, gap_extension_penalty=8):

        """
        Local sequence alignment with Smith-Waterman.

        Args:
            s1 (iter)
            s2 (iter)
        """

        self.matrix = np.zeros([len(s1)+1, len(s2)+1])

        self.pointers = {}

        for r in range(1, len(s1)+1):
            for c in range(1, len(s2)+1):

                # max of:

                # 1 - take the value of the diagonal neighbor (1 up, 1 left)
                #   - if the current cell is a match, add match score,
                #     otherwise subtract the mismatch score

                diagonal_score = (
                    self.matrix[r-1][c-1] +
                    (match_score if s1[r-1] == s2[c-1] else mismatch_score)
                )

                # 2 - greatest column gap score
                #   - take the values in the column above the current cell and
                #     get the max score
                #   - max score - (gap_penalty + (gap_ext_penalty * distance))

                col = self.matrix[:, c][:r]

                col_max_idx = np.argwhere(col == np.amax(col)).flatten()[-1]

                col_max_distance = len(col) - col_max_idx

                col_max_score = col[col_max_idx]

                col_score = (
                    col_max_score -
                    (gap_penalty + (gap_extension_penalty * col_max_distance))
                )

                # 3 - greatest row gap score - same for row

                row = self.matrix[r, :][:c]

                row_max_idx = np.argwhere(row == np.amax(row)).flatten()[-1]

                row_max_distance = len(row) - row_max_idx

                row_max_score = row[row_max_idx]

                row_score = (
                    row_max_score -
                    (gap_penalty + (gap_extension_penalty * row_max_distance))
                )

                # Compute the new score.
                score = max(
                    diagonal_score,
                    col_score,
                    row_score,
                    0,
                )

                # Update the backpointers.

                if score == diagonal_score:
                    self.pointers[r,c] = (r-1, c-1, True)

                elif score == col_score:
                    self.pointers[r,c] = (r-1, c, False)

                elif score == row_score:
                    self.pointers[r,c] = (r, c-1, False)

                else:
                    self.pointers[r,c] = False

                # Fill the cell.
                self.matrix[r,c] = score


    def extract(self):

        """
        Extract the optimum alignment.
        """

        pass
