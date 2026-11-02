# Games That Don't Die

Code and data for the paper **"Games That Don't Die: Operator-Independent Competitive Infrastructure for
Abandoned Multiplayer Games"** (Félix Francia, draft v0.2, November 2026).

When an online game loses its operator, communities can revive the servers, but the competitive memory
(rankings, match history, titles) dies with whoever kept the database. The paper asks how a ranking can
outlive any operator when every match is reported by a single, untrusted community server, and tests the
answer on the real match history of [Warband Premier](https://warbandpremier.com), a competitive network
for *Mount & Blade: Warband* (2010) that went silent in May 2026 and relaunches on 1 December 2026 as a
network of community-hosted rooms.

Everything in this repository runs from one file of anonymised real matches. Every table of results in the
paper can be regenerated with one command.

## What the paper shows

Proven (proofs in the paper, Sections 5 and 8):

1. **A single host's lies are unidentifiable** (Proposition 1). If a player only plays on one server, "the
   server lies for them" and "they are stronger" produce exactly the same data.
2. **Only relative honesty is observable** (Proposition 4). No method can measure whether a server is honest
   in absolute terms, only whether it treats a player differently from other servers.
3. **A per-player honest majority** (Proposition 5). Under the natural gauge, rewriting a player's rating
   requires controlling most of *that player's* matches: a rating analogue of the 51% assumption of
   blockchains, counted per player rather than over the whole network.
4. **Detection with a bounded number of matches** (Propositions 2 and 3, Lemma 1), including a model-free
   test whose threshold contains the overlap explicitly and which abstains instead of accusing when a
   player has no matches elsewhere.
5. **A fake-account farm's trust, and hence its tokens, is bounded by its contact with real players**
   (Propositions 6 and 7).

Measured on 1,435 real matches by 138 players, with simulated hosts and injected attacks (Section 9):

| Finding | Script | Output |
| --- | --- | --- |
| A lying host is caught in 89–100% of trials once it flips half of a target's losses; plain Elo lets the target climb 8–20 places. Without cross-host overlap, honest hosts of strong players are falsely accused in up to 44% of trials. | `src/srtd.py` | `results/lying_host.json` |
| Calibration of that test with no attack | `src/null_calib.py` | `results/null_calibration.txt` |
| A joint skill-and-bias model under a median-like gauge removes about two thirds of the rank a lie buys and flags no honest host; a mean gauge fails, as Proposition 5 predicts. | `src/joint.py` | `results/joint.json` |
| The model-free test makes no false accusation in 180 attack-free trials and is most powerful when a player splits matches evenly. | `src/modelfree.py` | `results/modelfree.json` |
| A farm of 12 fake accounts takes 17% of tokens under per-match issuance and 19% under information-only weighting, but at most 3.2% with trust; a trust gate keeps every fake account out of the ranking. | `src/sybil.py` | `results/sybil_*.json` |

Limits, stated plainly: the historical matches were all played on the operator's own servers, so hosts are
simulated; the data come from one game and one community; the extension of known Bradley–Terry error bounds
to team matches is an open step (Section 5.2).

## Reproduce

Requirements: Python 3.10+ and NumPy. Nothing else.

```bash
pip install numpy
./run_all.sh              # all experiments in parallel, about one hour on a laptop
./run_all.sh lying        # or one at a time: lying | null | joint | modelfree | sybil
```

Results are written to `results/`. All randomness is seeded, so runs are repeatable.

## Data

`data/matches.json` holds every decided match of Warband Premier's beta, Season 1 and Season 2
(20 February to 8 May 2026), in the order they were played:

```json
{"order": 0, "season": 1, "mode": "competitive_6v6", "winner": 2,
 "team1": ["p3f9a…", "…"], "team2": ["p81c2…", "…"]}
```

Player identifiers are random strings generated once at export time; the mapping to real accounts was never
written to disk. Timestamps, maps, scores, names and per-player statistics are deliberately left out.
Warband Premier's website shows match history publicly, so perfect anonymity cannot be guaranteed against
someone willing to reconstruct the sequence of results; the dataset adds nothing that is not already public
and removes everything the experiments do not need. `export/export_public.js` is the exact read-only script
that produced the file.

Thanks to the players of the Warband Premier community, whose matches made this research possible.

## License

Code: MIT (see `LICENSE`). Data: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Cite

```bibtex
@misc{francia2026gamesthatdontdie,
  title  = {Games That Don't Die: Operator-Independent Competitive Infrastructure for Abandoned Multiplayer Games},
  author = {Francia, F\'elix},
  year   = {2026},
  note   = {Draft v0.2}
}
```
