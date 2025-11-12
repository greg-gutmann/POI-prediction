# Reviewing Results (with concrete examples)

This guide shows how to review metrics, logs, and checkpoints from previous runs using the existing folders in this repo.

Example runs detected:

```
tensorboard/20251111_153849/nyc
tensorboard/20251111_143142/nyc
tensorboard/20251111_130942/nyc
```

Pick the latest (20251111_153849/nyc) for the examples below.

## 1) Inspect checkpoint and TensorBoard files

- List files in the run directory
  ```bash
  ls -l tensorboard/20251111_153849/nyc
  ```
  Expect to see:
  - `checkpoint.pt`
  - `events.out.tfevents...` (TensorBoard scalar logs)

## 2) Read the text log for metrics

Logs are stored separately under `log/<timestamp>/<dataset>/train.log`.

- Show the tail of the log
  ```bash
  tail -n 60 log/20251111_153849/nyc/train.log
  ```

- Extract the final test summary (saved when `do_test` ran)
  ```bash
  grep -n "Test evaluation result" log/20251111_153849/nyc/train.log
  ```
  Example line structure:
  ```
  [Evaluating] Test evaluation result : {'hparam/num_params': ..., 'hparam/Recall@1': ..., 'hparam/Recall@5': ..., 'hparam/Recall@10': ..., 'hparam/Recall@20': ..., 'hparam/NDCG@K': ..., 'hparam/MAP@K': ..., 'hparam/MRR': ...}
  ```

- During training, validation/test snapshots are also logged periodically. To view:
  ```bash
  grep -n "[Evaluating] Recall@" log/20251111_153849/nyc/train.log | tail -n 10
  ```

## 3) Visualize in TensorBoard

- Open TensorBoard scoped to this run or the whole directory
  ```bash
  tensorboard --logdir tensorboard/20251111_153849/nyc
  # or
  tensorboard --logdir tensorboard
  ```
  Useful tags:
  - `train/loss_step`, `train/loss_epoch`
  - `validate/Recall@K`, `validate/MRR`, `validate/eval_loss`
  - `test/Recall@K`, `test/MRR`, `test/eval_loss`
  - `hparams` (summary table at the end)

## 4) Re-evaluate a checkpoint today (no retraining)

Use the provided test runner. If `run_args.init_checkpoint` is unset, it auto-picks the latest checkpoint for the dataset.

```bash
python run_test.py -f best_conf/nyc.yml
```
This will write a new log under `log/<new_timestamp>/nyc/train.log` and print metrics.

## 5) Aggregate sanity across recent runs

`compute_acc.py` parses the last "Test evaluation result" lines and prints/records min/mean/max:

```bash
python compute_acc.py
# (defaults to dataset='nyc' in __main__)
```
Outputs include arrays of Recall@K and MRR, and a summary written to `acc.txt`.

## 6) Per-class metrics and raw predictions (optional)

Use the cold-start evaluation script to write detailed CSVs:

```bash
python run_cold_start.py -f best_conf/nyc.yml
```
Artifacts:
- `log/<timestamp>/nyc/class_metrics.csv` (columns: valid_indices, class_correct, class_total, class_accuracy)
- `log/<timestamp>/nyc/test_predictions.csv` (per-row top-k predictions and scores)

Quick peeks:
```bash
head -n 5 log/20251111_153849/nyc/class_metrics.csv
head -n 5 log/20251111_153849/nyc/test_predictions.csv
```

## 7) Quick sanity checks

- Number of POIs (for random-baseline intuition):
  ```bash
  grep -n "[Initialize Dataset] #poi:" log/20251111_153849/nyc/train.log
  ```
  A purely random top-K predictor has `Recall@K ≈ K / #poi` (should be very small). Your model should exceed this by a wide margin.

- Popularity bias check:
  - In `class_metrics.csv`, ensure accuracy isn’t only high for a handful of classes and near-zero for the rest.
  - In `test_predictions.csv`, spot-check that predictions aren’t a short list of the same POIs for everyone.

## 8) Compare runs

Repeat the commands above for:
```
tensorboard/20251111_143142/nyc
log/20251111_143142/nyc/train.log

tensorboard/20251111_130942/nyc
log/20251111_130942/nyc/train.log
```
This lets you confirm that improvements are consistent across timestamps (lower variance is better).

---

If you want, I can surface the exact final Test evaluation lines for these timestamps or open TensorBoard for you; tell me which run(s) to inspect.
