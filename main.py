# Interactive entry point for cricket matches (human vs human or human vs AI).

import random
import sys
from pathlib import Path
from time import sleep

sys.path.insert(0, str(Path(__file__).resolve().parent))

from typing import List

from agents.dqn_agent import DQNAgent
from agents.q_learning_agent import QLearningAgent
from environment.drs_system import DRSSystem
from environment.probability_engine import (
    ALL_DELIVERIES,
    FAST_DELIVERIES,
    SPIN_DELIVERIES,
    ProbabilityEngine,
)
from models.player import Player
from models.team import Team
from utils.helpers import (
    format_overs,
    get_int_input,
    get_str_input,
    parse_runs,
    short_form_result,
)

DISMISSAL_TYPES = {
    "L.B.W",
    "Bowled",
    "Caught",
    "Caught and Bowled",
    "Run Out",
    "Stumped",
    "Edged And Caught Behind",
}

RUN_MAP = {"No Run": 0, "1 Run": 1, "2 Runs": 2, "3 Runs": 3, "4 Runs": 4, "6 Runs": 6}


# Human Match Engine


class HumanMatch:
    # Interactive match simulation supporting both human and AI players.

    def __init__(
        self,
        overs: int,
        home_team: Team,
        away_team: Team,
        batter_agent=None,
        bowler_agent=None,
        ai_is_home: bool = None,
    ):
        self.overs = overs
        self.total_balls = overs * 6
        self.home_team = home_team
        self.away_team = away_team
        self.match_format = "T20"

        self._engine = ProbabilityEngine()
        self._drs = DRSSystem()

        self.batter_agent = batter_agent
        self.bowler_agent = bowler_agent
        self.ai_is_home = ai_is_home
        self._human_team: Team = None
        self._ai_team: Team = None
        self.batting_team: Team = None
        self.bowling_team: Team = None
        self.striker = None
        self.non_striker = None
        self.balls_bowled: int = 0
        self.overs_bowled: int = 0
        self.innings: int = 0
        self.powerplay: bool = True
        self.target = None
        self.super_over: bool = False

        self.delivery: str = ""
        self.stumps: str = ""
        self.shot: str = ""
        self.result: str = ""
        self.over_log = []
        self.used_bowlers = []

    def _start_innings(self) -> None:
        self.balls_bowled = 0
        self.overs_bowled = 0
        self.powerplay = True
        self.over_log = []
        self.used_bowlers = []

        self.batting_team.score = 0
        self.batting_team.wickets = 0
        self.batting_team.reviews_left = 2
        self.bowling_team.reviews_left = 2
        self.bowling_team.current_bowler = None

        for player in self.batting_team.players + self.bowling_team.players:
            player.reset()

        self.striker = self.batting_team.players[0]
        self.non_striker = self.batting_team.players[1]

    def _swap_strike(self) -> None:
        self.striker, self.non_striker = self.non_striker, self.striker

    def toss(self) -> None:
        print(
            f"\nCaptains: {self.home_team.captain} (home) vs {self.away_team.captain} (away)"
        )
        sleep(1)
        call = get_str_input(
            f"{self.home_team.name}, call the toss (Heads/Tails): ",
            ["Heads", "Tails"],
        )
        print("Tossing the coin ...")
        sleep(2)
        result = random.choice(["Heads", "Tails"])
        print(f"Coin shows: {result}")
        sleep(1)

        winner = self.home_team if call == result else self.away_team
        print(f"\n{winner.name} won the toss!")
        choice = get_str_input(f"{winner.name}, choose Bat or Bowl: ", ["Bat", "Bowl"])
        if choice == "Bat":
            self.batting_team, self.bowling_team = winner, (
                self.away_team if winner == self.home_team else self.home_team
            )
        else:
            self.bowling_team, self.batting_team = winner, (
                self.away_team if winner == self.home_team else self.home_team
            )

        if self.ai_is_home is not None:
            if self.ai_is_home:
                self._ai_team = self.home_team
                self._human_team = self.away_team
            else:
                self._human_team = self.home_team
                self._ai_team = self.away_team

    def _select_bowler(self) -> None:
        if self.bowling_team == self._ai_team and self.bowler_agent is not None:
            bowlers = self.bowling_team.fast_bowlers + self.bowling_team.spin_bowlers
            options = [b for b in bowlers if b != self.bowling_team.current_bowler_name]
            choice = random.randint(0, len(options) - 1)
            self.bowling_team.current_bowler_name = options[choice]
            # Find existing player from roster instead of creating new one
            self.bowling_team.current_bowler = next(
                (
                    p
                    for p in self.bowling_team.players
                    if p.name == self.bowling_team.current_bowler_name
                ),
                None,
            )
            print(f"\nAI selects bowler: {self.bowling_team.current_bowler_name}")
            return

        bowlers = self.bowling_team.fast_bowlers + self.bowling_team.spin_bowlers
        options = [b for b in bowlers if b != self.bowling_team.current_bowler_name]
        print("\nAvailable Bowlers:")
        for i, name in enumerate(options, 1):
            print(f"  {i}. {name}")
        choice = (
            get_int_input(f"Choose Bowler (1-{len(options)}): ", 1, len(options)) - 1
        )
        self.bowling_team.current_bowler_name = options[choice]
        # Find existing player from roster instead of creating new one
        self.bowling_team.current_bowler = next(
            (
                p
                for p in self.bowling_team.players
                if p.name == self.bowling_team.current_bowler_name
            ),
            None,
        )
        print(f"\n{self.bowling_team.current_bowler_name} will bowl this over.")
        sleep(1)

    def _compute_batter_state(self) -> List[float]:
        delivery_idx = (
            ALL_DELIVERIES.index(self.delivery)
            if self.delivery in ALL_DELIVERIES
            else 0
        )
        stumps_idx = 1 if self.stumps == "Touching" else 0
        wickets_remaining = 10 - self.batting_team.wickets
        balls_remaining = self.total_balls - self.balls_bowled
        runs_required = (
            float(self.target - self.batting_team.score)
            if self.target is not None
            else 0.0
        )
        overs_played = self.balls_bowled / 6.0 if self.balls_bowled > 0 else 0.0
        current_rr = (
            min(36.0, self.batting_team.score / overs_played)
            if overs_played > 0
            else 0.0
        )
        required_rr = 0.0
        if self.target is not None and balls_remaining > 0:
            required_rr = min(36.0, runs_required / (balls_remaining / 6.0))
        return [
            float(delivery_idx),
            float(stumps_idx),
            float(wickets_remaining),
            float(balls_remaining),
            runs_required,
            current_rr,
            required_rr,
        ]

    def _compute_bowler_state(self) -> List[float]:
        delivery_idx = (
            ALL_DELIVERIES.index(self.delivery)
            if self.delivery in ALL_DELIVERIES
            else 0
        )
        stumps_idx = 1 if self.stumps == "Touching" else 0
        wickets_remaining = 10 - self.batting_team.wickets
        balls_remaining = self.total_balls - self.balls_bowled
        runs_required = (
            float(self.target - self.batting_team.score)
            if self.target is not None
            else 0.0
        )
        overs_played = self.balls_bowled / 6.0 if self.balls_bowled > 0 else 0.0
        current_rr = (
            min(36.0, self.batting_team.score / overs_played)
            if overs_played > 0
            else 0.0
        )
        required_rr = 0.0
        if self.target is not None and balls_remaining > 0:
            required_rr = min(36.0, runs_required / (balls_remaining / 6.0))
        return [
            float(delivery_idx),
            float(stumps_idx),
            float(wickets_remaining),
            float(balls_remaining),
            runs_required,
            current_rr,
            required_rr,
        ]

    def _select_shot(self) -> None:
        shots = self._engine.get_available_shots(self.delivery, self.stumps)
        if self.batting_team == self._ai_team and self.batter_agent is not None:
            state = self._compute_batter_state()
            avail = list(range(len(shots)))
            action = self.batter_agent.choose_action(state, avail)
            self.shot = shots[action]
            print(f"\nAgent selects shot: {self.shot}")
            return

        print("\nAvailable Shots:")
        for i, s in enumerate(shots, 1):
            print(f"  {i}. {s}")
        choice = get_int_input(f"Choose Shot (1-{len(shots)}): ", 1, len(shots)) - 1
        self.shot = shots[choice]

    def _select_delivery(self) -> None:
        deliveries = (
            FAST_DELIVERIES
            if self.bowling_team.current_bowler_name in self.bowling_team.fast_bowlers
            else SPIN_DELIVERIES
        )
        if self.bowling_team == self._ai_team and self.bowler_agent is not None:
            state = self._compute_bowler_state()
            avail = list(range(len(deliveries)))
            action = self.bowler_agent.choose_action(state, avail)
            self.delivery = deliveries[action]
            print(f"\nBowler agent selects delivery: {self.delivery}")
            return

        print("\nAvailable Deliveries:")
        for i, d in enumerate(deliveries, 1):
            print(f"  {i}. {d}")
        choice = (
            get_int_input(
                f"Choose Delivery (1-{len(deliveries)}): ", 1, len(deliveries)
            )
            - 1
        )
        self.delivery = deliveries[choice]

    def _handle_drs(self, result: str) -> str:
        reviewable_batting = {"L.B.W", "Run Out", "Edged And Caught Behind", "Stumped"}
        if result in reviewable_batting and self.batting_team.reviews_left > 0:
            ans = input("Take a DRS review? (yes/no): ").strip().lower()
            if ans == "yes":
                final, success = self._drs.review(result, self.shot, verbose=True)
                if not success:
                    self.batting_team.reviews_left -= 1
                    print(f"Reviews left (batting): {self.batting_team.reviews_left}")
                return final
        elif result == "Leg Bye" and self.bowling_team.reviews_left > 0:
            ans = input("Bowling team review for LBW? (yes/no): ").strip().lower()
            if ans == "yes":
                final, success = self._drs.review("Leg Bye", self.shot, verbose=True)
                if not success:
                    self.bowling_team.reviews_left -= 1
                    print(f"Reviews left (bowling): {self.bowling_team.reviews_left}")
                return final
        return result

    def _bowl_ball(self, wickets_disabled: bool = False) -> None:
        self._select_delivery()
        self.stumps = random.choice(["Touching", "Not Touching"])
        self._select_shot()

        # Sample extras
        extra = self._engine.sample_extras()
        if extra != "None":
            self.result = extra
        else:
            self.result = self._engine.sample_outcome(
                self.shot,
                self.delivery,
                self.stumps,
                self.match_format,
                self.powerplay,
            )

        # Apply game rules
        if self.shot == "Leave":
            self.result = "No Run"
        if self.result == "L.B.W" and self.stumps == "Not Touching":
            self.result = "Leg Bye"
        while self.result == "Bowled" and self.stumps == "Not Touching":
            self.result = self._engine.sample_outcome(
                self.shot,
                self.delivery,
                self.stumps,
                self.match_format,
                self.powerplay,
            )
        while self.result in ("Wide", "Wide Four") and self.stumps == "Touching":
            self.result = self._engine.sample_outcome(
                self.shot,
                self.delivery,
                self.stumps,
                self.match_format,
                self.powerplay,
            )

        catch_drop = False
        if self.result in ("Caught", "Caught and Bowled", "Edged And Caught Behind"):
            if random.random() < 0.15:
                self.result = random.choice(["No Run", "1 Run", "2 Runs"])
                catch_drop = True

        if self.result in DISMISSAL_TYPES and wickets_disabled:
            self.result = random.choices(
                ["No Run", "1 Run", "2 Runs", "3 Runs", "4 Runs", "6 Runs"],
                weights=[30, 40, 20, 2, 6, 1],
            )[0]

        self.result = self._handle_drs(self.result)

        print(
            f"\n{self.bowling_team.current_bowler_name} → {self.striker.name}: {self.delivery}"
        )
        sleep(0.5)
        print(f"Stumps: {self.stumps}")
        sleep(0.5)
        if catch_drop:
            print("  *** Catch Dropped! ***")
            sleep(0.5)
        print(f"Result: {self.result}")
        sleep(0.8)
        if self.bowling_team.current_bowler is None:
            print("⚠️ ERROR: No bowler selected! Skipping ball.")
            return

        if self.result == "Wide":
            self.batting_team.score += 1
            self.bowling_team.current_bowler.runs_conceded += 1
            self.over_log.append("WD")
            return
        if self.result == "Wide Four":
            self.batting_team.score += 5
            self.bowling_team.current_bowler.runs_conceded += 5
            self.over_log.append("5WD")
            return
        if self.result == "Leg Bye":
            self.batting_team.score += 1
            self.balls_bowled += 1
            self.bowling_team.current_bowler.balls_bowled += 1
            self.over_log.append("LB")
            return
        if self.result == "No Ball":
            self.batting_team.score += 1
            self.bowling_team.current_bowler.runs_conceded += 1
            self.over_log.append("NB")
            return

        if self.result in DISMISSAL_TYPES:
            self.striker.balls += 1
            self.striker.dismissed = True
            self.striker.how_out = self.result
            self.striker.wicket_taking_bowler_name = (
                self.bowling_team.current_bowler_name
            )
            self.bowling_team.current_bowler.wickets += 1
            self.batting_team.wickets += 1
            self.over_log.append("W")
            print(f"  OUT! {self.striker.name} — {self.result}")
            next_bat = self.batting_team.get_next_batsman()
            if next_bat:
                self.striker = next_bat
        else:
            runs = parse_runs(self.result)
            self.striker.add_runs(runs)
            if runs == 4:
                self.striker.fours += 1
            elif runs == 6:
                self.striker.sixes += 1
            self.batting_team.score += runs
            self.bowling_team.current_bowler.runs_conceded += runs
            self.over_log.append(short_form_result(self.result))
            if runs % 2 == 1:
                self._swap_strike()

        self.balls_bowled += 1
        self.bowling_team.current_bowler.balls_bowled += 1

    # Score display and over summary methods

    def _display_score(self) -> None:
        print(
            f"\n  {self.batting_team.name}: {self.batting_team.score}/{self.batting_team.wickets}"
        )
        print(f"  {self.striker}  |  {self.non_striker}")
        if self.balls_bowled > 0:
            rr = round(self.batting_team.score / (self.balls_bowled / 6), 2)
            print(f"  RR: {rr}  Overs: {format_overs(self.balls_bowled)}")
        if self.innings in (2, 4) and self.target:
            need = self.target - self.batting_team.score
            left = self.total_balls - self.balls_bowled
            print(f"  Need {need} from {left} balls")

    def _print_over_summary(self) -> None:
        print(
            f"\n  Over summary ({self.bowling_team.current_bowler_name}): {' | '.join(self.over_log)}"
        )

    def play_innings(self) -> None:
        self._start_innings()
        self._display_score()

        while self.balls_bowled < self.total_balls:
            self.overs_bowled += 1
            self._select_bowler()

            over_start = self.balls_bowled
            while self.balls_bowled < over_start + 6:
                self._bowl_ball()
                self._display_score()

                # Check innings end conditions
                if self.batting_team.wickets == 10:
                    print("All Out!")
                    return
                if (
                    self.innings in (2, 4)
                    and self.target
                    and self.batting_team.score >= self.target
                ):
                    print("Target chased down!")
                    return

            # End of over
            self._print_over_summary()
            self.over_log = []
            self._swap_strike()

            # Powerplay logic
            overs_done = self.balls_bowled // 6
            if self.match_format == "T20" and overs_done >= 6 and self.powerplay:
                self.powerplay = False
                print("  *** Powerplay Over! ***")
            elif self.match_format == "ODI" and overs_done >= 10 and self.powerplay:
                self.powerplay = False
                print("  *** Powerplay Over! ***")

            input("Press Enter to continue...")
            sleep(0.5)

    def _man_of_the_match(self) -> None:
        all_players = self.home_team.players + self.away_team.players
        best, best_score = None, float("-inf")
        for p in all_players:
            if p.balls_bowled == 0 and p.balls == 0:
                continue
            economy = (
                round(p.runs_conceded / (p.balls_bowled / 6), 2)
                if p.balls_bowled >= 6
                else (p.runs_conceded if p.balls_bowled > 0 else 0)
            )
            impact = (p.runs - p.balls / 2) + (p.wickets * 25 - economy * 2)
            if impact > best_score:
                best_score, best = impact, p
        if best:
            print(f"\n🏆 Man of the Match: {best.name}")
            print(
                f"   Batting: {best.runs} off {best.balls}  |  Bowling: {best.wickets}W / {best.runs_conceded}R"
            )
        else:
            print("\n(No standout performer this match)")

    def play_match(self, match_format: str = "T20") -> None:
        self.match_format = match_format.upper()
        self.toss()
        self.innings = 1

        input("\nPress Enter to start First Innings...")
        sleep(1)
        print("\n========== First Innings ==========\n")
        self.play_innings()
        self.batting_team.print_scorecard()
        self.bowling_team.print_bowler_scorecard()

        self.target = self.batting_team.score + 1
        self.innings = 2
        self.batting_team, self.bowling_team = self.bowling_team, self.batting_team

        print(f"\nTarget for {self.batting_team.name}: {self.target}")
        input("Press Enter to start Second Innings...")
        sleep(1)
        print("\n========== Second Innings ==========\n")
        self.play_innings()
        self.batting_team.print_scorecard()
        self.bowling_team.print_bowler_scorecard()

        print("\n=== Match Over ===")
        ht, at = self.home_team, self.away_team
        print(
            f"{ht.name}: {ht.score}/{ht.wickets}  vs  {at.name}: {at.score}/{at.wickets}"
        )

        if self.bowling_team.score > self.batting_team.score:
            margin = self.bowling_team.score - self.batting_team.score
            print(f"{self.bowling_team.name} won by {margin} runs.")
        elif self.batting_team.score > self.bowling_team.score:
            wickets_left = 10 - self.batting_team.wickets
            print(f"{self.batting_team.name} won by {wickets_left} wickets.")
        else:
            print("Match Tied!")
            if self.match_format == "T20":
                self._play_super_over()
                return

        self._man_of_the_match()

    def _play_super_over(self) -> None:
        print("\n--- Scores level! Super Over begins. ---")
        sleep(1)
        self.super_over = True
        self.total_balls = 6
        self.innings = 2

        # Team 1 bats
        input(f"Press Enter for {self.batting_team.name} Super Over...")
        self.play_innings()
        t1_score = self.batting_team.score

        # Team 2 bats
        self.batting_team, self.bowling_team = self.bowling_team, self.batting_team
        self.target = t1_score + 1
        input(
            f"Press Enter for {self.batting_team.name} Super Over (target {self.target})..."
        )
        self.play_innings()
        t2_score = self.batting_team.score

        if t2_score > t1_score:
            print(f"\n{self.batting_team.name} wins the Super Over!")
        elif t2_score < t1_score:
            print(f"\n{self.bowling_team.name} wins the Super Over!")
        else:
            print("\nSuper Over tied too! Another Super Over ...")
            self._play_super_over()


def setup_team(team_name: str) -> Team:
    captain = input(f"Enter {team_name} Captain: ").strip()
    team = Team(team_name, captain)

    print(f"\nEnter 11 players for {team_name}:")
    player_names = []
    for i in range(11):
        name = input(f"  Player {i+1}: ").strip()
        player_names.append(name)
        team.add_player(Player(name))

    if captain not in player_names:
        print(
            f"⚠ Captain '{captain}' not in players list! Setting captain to first player."
        )
        team.captain = player_names[0]

    fast_bowlers = set()
    spin_bowlers = set()

    fast_count = get_int_input(
        f"\nHow many fast bowlers for {team_name}? (1-6): ", 1, 6
    )
    for i in range(fast_count):
        while True:
            bowler_name = input(f"  Fast bowler {i+1} (must be a player): ").strip()
            if bowler_name not in player_names:
                print(f"  ⚠ '{bowler_name}' not in players list. Try again.")
            elif bowler_name in spin_bowlers:
                print(
                    f"  ⚠ '{bowler_name}' is already a spin bowler. Try another player."
                )
            else:
                team.add_fast_bowler(bowler_name)
                fast_bowlers.add(bowler_name)
                break

    spin_count = get_int_input(
        f"\nHow many spin bowlers for {team_name}? (1-6): ", 1, 6
    )
    for i in range(spin_count):
        while True:
            bowler_name = input(f"  Spin bowler {i+1} (must be a player): ").strip()
            if bowler_name not in player_names:
                print(f"  ⚠ '{bowler_name}' not in players list. Try again.")
            elif bowler_name in fast_bowlers:
                print(
                    f"  ⚠ '{bowler_name}' is already a fast bowler. Try another player."
                )
            else:
                team.add_spin_bowler(bowler_name)
                spin_bowlers.add(bowler_name)
                break

    return team


if __name__ == "__main__":
    print("=" * 55)
    print("  Welcome to Cricket AI Simulator — Interactive Mode")
    print("=" * 55)
    sleep(1)

    # Mode selection: human vs human or human vs AI
    mode = get_str_input(
        "\nGame Mode (Human vs Human / Human vs AI): ",
        ["Human vs Human", "Human vs AI"],
    )
    is_ai_mode = mode == "Human vs AI"

    match_format = get_str_input(
        "\nMatch format (T20 / ODI / Test): ", ["T20", "ODI", "Test"]
    )
    overs = get_int_input("\nNumber of overs per innings: ", 1, 100)

    # Team names
    t1_name = input("Home team name: ").strip()
    t2_name = input("Away team name: ").strip()

    # Set up teams
    home_team = setup_team(t1_name)
    away_team = setup_team(t2_name)

    batter_agent = None
    bowler_agent = None
    ai_is_home = False

    if is_ai_mode:
        ai_team_choice = get_str_input(
            f"Which team will be AI? ({t1_name} / {t2_name}): ", [t1_name, t2_name]
        )
        ai_is_home = ai_team_choice == t1_name

        print(f"\n[Loading AI agents for {ai_team_choice}...]")

        # Try to load separate agent checkpoints (preferred)
        batter_checkpoint = "training/checkpoints/batter_agent.pkl"
        bowler_checkpoint = "training/checkpoints/bowler_agent.pkl"
        fallback_checkpoint = "training/checkpoints/qlearning_final.pkl"

        # Load batter agent
        try:
            batter_agent = QLearningAgent.load(batter_checkpoint)
            print(f"✓ Loaded batter agent from {batter_checkpoint}")
        except FileNotFoundError:
            print(f"⚠ Batter checkpoint not found at {batter_checkpoint}")
            try:
                batter_agent = QLearningAgent.load(fallback_checkpoint)
                print(f"✓ Loaded batter agent from {fallback_checkpoint} (fallback)")
            except FileNotFoundError:
                print(f"✗ No batter agent available")
                batter_agent = None

        # Load bowler agent
        try:
            bowler_agent = QLearningAgent.load(bowler_checkpoint)
            print(f"✓ Loaded bowler agent from {bowler_checkpoint}")
        except FileNotFoundError:
            print(f"⚠ Bowler checkpoint not found at {bowler_checkpoint}")
            try:
                bowler_agent = QLearningAgent.load(fallback_checkpoint)
                print(f"✓ Loaded bowler agent from {fallback_checkpoint} (fallback)")
            except FileNotFoundError:
                print(f"✗ No bowler agent available, AI will bat-only")
                bowler_agent = None

    match = HumanMatch(
        overs,
        home_team,
        away_team,
        batter_agent=batter_agent,
        bowler_agent=bowler_agent,
        ai_is_home=ai_is_home if is_ai_mode else None,
    )
    match.play_match(match_format)
