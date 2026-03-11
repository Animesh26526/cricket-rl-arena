"""
environment/drs_system.py
--------------------------
Handles the Decision Review System (DRS) logic.
All probabilistic review outcomes are isolated here so the main
match engine stays clean.
"""

import random
from typing import Tuple


class DRSSystem:
    """
    Simulates TV-umpire / third-umpire review decisions.

    Methods
    -------
    review(dismissal_type, shot) -> Tuple[str, bool]
        Returns (final_result, successful_review).
        `final_result` is either the original dismissal or "Not Out".
        `successful_review` is True when the batting/bowling team wins.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def review(
        self, dismissal_type: str, shot: str, verbose: bool = False
    ) -> Tuple[str, bool]:
        """
        Simulate a DRS review.

        Parameters
        ----------
        dismissal_type : str
            The on-field decision being reviewed.
        shot : str
            The shot the batter played (needed for LBW logic).
        verbose : bool
            Print review commentary (used during human play).

        Returns
        -------
        Tuple[str, bool]
            (final_result, successful_review)
        """
        handler = {
            "L.B.W": self._review_lbw,
            "Leg Bye": self._review_leg_bye,
            "Run Out": self._review_run_out,
            "Edged And Caught Behind": self._review_edge_catch,
            "Stumped": self._review_stumped,
        }
        func = handler.get(dismissal_type)
        if func is None:
            if verbose:
                print("Review system not applicable to this type.")
            return dismissal_type, False

        return func(shot, verbose)

    # ------------------------------------------------------------------
    # Private review handlers
    # ------------------------------------------------------------------

    def _review_lbw(self, shot: str, verbose: bool) -> Tuple[str, bool]:
        edge = random.choices(["No Bat", "Bat"], [0.95, 0.05])[0]
        pitching = random.choices(
            ["In Line", "Outside Off", "Outside Leg"], [0.6, 0.25, 0.15]
        )[0]
        impact = random.choices(
            ["In Line", "Umpire's Call", "Outside"], [0.7, 0.2, 0.1]
        )[0]
        wickets = random.choices(
            ["Hitting", "Umpire's Call", "Missing"], [0.6, 0.25, 0.15]
        )[0]

        if verbose:
            print(f"UltraEdge: {edge}")
            print(f"Pitching: {pitching}  Impact: {impact}  Wickets: {wickets}")

        if (
            edge == "Bat"
            or pitching == "Outside Leg"
            or impact == "Outside"
            or wickets == "Missing"
        ):
            if verbose:
                print("Decision Overturned! Not Out.")
            return "Not Out", True

        if verbose:
            print("Umpire's Call Stands. Batter is Out.")
        return "L.B.W", False

    def _review_leg_bye(self, shot: str, verbose: bool) -> Tuple[str, bool]:
        edge = random.choices(["No Bat", "Bat"], [0.98, 0.02])[0]
        pitching = random.choices(
            ["In Line", "Outside Off", "Outside Leg"], [0.3, 0.4, 0.3]
        )[0]
        impact = random.choices(
            ["In Line", "Umpire's Call", "Outside"], [0.3, 0.4, 0.3]
        )[0]
        wickets = random.choices(
            ["Hitting", "Umpire's Call", "Missing"], [0.3, 0.2, 0.5]
        )[0]

        if verbose:
            print(f"UltraEdge: {edge}")
            print(f"Pitching: {pitching}  Impact: {impact}  Wickets: {wickets}")

        hitting_inline = (
            pitching == "In Line" and impact == "In Line" and wickets == "Hitting"
        )
        hitting_leave = (
            pitching == "Outside Off"
            and impact == "Outside"
            and shot == "Leave"
            and wickets == "Hitting"
        )
        if edge == "No Bat" and (hitting_inline or hitting_leave):
            if verbose:
                print("Review Successful. Changed to L.B.W.")
            return "L.B.W", True

        if verbose:
            print("Review Unsuccessful. Stays Leg Bye.")
        return "Leg Bye", False

    def _review_run_out(self, shot: str, verbose: bool) -> Tuple[str, bool]:
        decision = random.choices(["Out", "Not Out"], [0.8, 0.2])[0]
        if verbose:
            print(f"Third Umpire View: {decision}")
        if decision == "Not Out":
            if verbose:
                print("Review Successful. Overturned to Not Out.")
            return "Not Out", True
        if verbose:
            print("Review Unsuccessful. Run Out stands.")
        return "Run Out", False

    def _review_edge_catch(self, shot: str, verbose: bool) -> Tuple[str, bool]:
        edge = random.choices(["No Edge", "Edge"], [0.2, 0.8])[0]
        if verbose:
            print(f"UltraEdge shows: {edge}")
        if edge == "No Edge":
            if verbose:
                print("Review Successful. No contact. Not Out.")
            return "Not Out", True
        if verbose:
            print("Review Unsuccessful. Out stands.")
        return "Edged And Caught Behind", False

    def _review_stumped(self, shot: str, verbose: bool) -> Tuple[str, bool]:
        foot = random.choices(["Foot In", "Foot Out"], [0.2, 0.8])[0]
        if verbose:
            print(f"Replay shows: {foot}")
        if foot == "Foot In":
            if verbose:
                print("Review Successful. Batter was inside. Not Out.")
            return "Not Out", True
        if verbose:
            print("Review Unsuccessful. Stumped remains.")
        return "Stumped", False
