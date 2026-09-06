Artefacts written by run_pipeline.py and experiments_extra.py.

  results.json / results_extra.json   every number reported in the paper
  table*.csv                          the paper's tables
  architecture.svg / .png, roc.png    the paper's figures
  adshield_model.joblib               trained bundle the Flask app loads
  demo_model.json                     distilled surrogate for the HTML explainer
  sample_test_clicks.csv              a small log to try the console's upload

adshield.db is created by the Flask app on first run and is git-ignored.
Pipeline checkpoints (_state.joblib and friends) are also ignored — they are
large and regenerable.
