# Wolfson Scheduler

![banner](banner.svg)

The Wolfson Scheduler is a tool to automate the weekly schedule drafting process for boat-clubs, by taking the rower's and coaches preferences into account.
This page discusses the technical details and provides a guide how to run a local copy of the server.
For general information see the [Main Page](https://botcs.github.io/wolfson-scheduler/).

## How to set up your own scheduler?
In order to run the algorithm you will need to do the following steps:
1. Set up the **Google Sheet API credentials**, so that the scheduler can access the Sheet data automatically. You can find a step-by-step guide to get going with the minimal requirements [here](GOOGLE_API_STEPS.md)
2. Once you have generated the `service_account.json` file and set up permissions to access the sheets, install the dependencies and launch the server.

```
git clone git@github.com:botcs/wolfson-scheduler.git
cd wolfson-scheduler

# Install dependencies
conda create -f requirements.yml
conda activate scheduler

# Launch server
TARGET_URL=<URL of your copy of the scheduler Google Sheets>

python server.py --target_spreadsheet_id=TARGET_URL --service_account_json="./service_account.json"
```

## Components
The scheduler consists of two main components:
1. `server.py` - Implements communication between the Google sheet of availabilities and the scheduler algorithm
2. `solver.py` - Implements the stateless scheduling algorithm.

## Workflow
The workflow looks as follows:
1. The `server` establishes connection to the sheets through the GSpread API
2. The names, preferences and availabilities of rowers are parsed from the target Google Sheets
3. The data is normalized and passed through sanity checks to catch irregularities (Errors are printed on the target Sheet)
4. The `solver` generates proposals in JSON format
5. The result is parsed by the `server` and printed on the target Sheet.

## Technical details
The `solver` implements a novel algorithm (please reach out if you have found something similar in the literature) to generate solutions *in parallel* for each outing.
This allows the tool to handle ~100 rowers and 5-10 boats a day reliably.

Once the outing proposals are ready, the weekly proposals are generated as an outer product of the outing proposals.
For each possible combination the scores are evaluated in parallel and are re-weighted using the user provided parameters.

From multiple evaluations the best one is stored and new alternatives are generated by permuting (making small variations to) the best proposal. This is repeated for a fixed number of iterations to increase the quality of the proposal.

## Contributions
If you find this work helpful and you would like to extend it, please do so by raising a GitHub issue first, then I would be happy to review your Pull Request.

The unit tests can be found in the `test.py` and the contributions are expected to pass the unit-test flawlessly before considered for review.
You can run the unit-tests by:
```
python -m unittest discover tests
```

