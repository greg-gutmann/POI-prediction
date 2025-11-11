## TL;DR: Single-user prediction (predict_user.py)

- **Basic (latest sample, auto-latest checkpoint)**

```bash
python predict_user.py --user_id 123 -f best_conf/nyc.yml
```

- **Predict all test samples for the user**

```bash
python predict_user.py --user_id 123 -f best_conf/nyc.yml --all
```

- **Select GPU explicitly**

```bash
python predict_user.py --user_id 123 -f best_conf/nyc.yml --gpu 0
```

- **Find an encoded user id (top 5 by frequency in test set)**

```bash
python -c "import pandas as pd; df=pd.read_csv('data/nyc/preprocessed/test_sample.csv'); print(df['UserId'].value_counts().head())"
```

Notes:
- The script expects the encoded `UserId` from `data/<dataset>/preprocessed/test_sample.csv`.
- If `--checkpoint` is not provided, it automatically uses the most recent `tensorboard/<timestamp>/<dataset>/checkpoint.pt`.
