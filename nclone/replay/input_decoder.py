#!/usr/bin/env python3
"""
Input decoding utilities for N++ binary replay parsing.

Handles the conversion of raw N++ input values to horizontal and jump components
using the original ntrace.py mapping.
"""

from typing import List, Tuple


class InputDecoder:
    """
    Decodes raw N++ input values to horizontal and jump components.
    
    Uses the original mapping from ntrace.py for compatibility with
    the original replay format.
    """
    
    # Input encoding dictionaries - ORIGINAL NTRACE.PY MAPPING!
    # Found the original mapping from ntrace.py - let's try this!
    # This is the mapping that was designed to work with the original replay format
    HOR_INPUTS_DIC = {0: 0, 1: 0, 2: 1, 3: 1, 4: -1, 5: -1, 6: -1, 7: -1}
    JUMP_INPUTS_DIC = {0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1, 6: 0, 7: 1}
    
    def decode_inputs(self, raw_inputs: List[int]) -> Tuple[List[int], List[int]]:
        """
        Decode raw input values to horizontal and jump components.

        Args:
            raw_inputs: List of raw input values (0-7)

        Returns:
            Tuple of (horizontal_inputs, jump_inputs)
        """
        hor_inputs = [self.HOR_INPUTS_DIC[inp] for inp in raw_inputs]
        jump_inputs = [self.JUMP_INPUTS_DIC[inp] for inp in raw_inputs]
        return hor_inputs, jump_inputs
