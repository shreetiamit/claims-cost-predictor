# notebooks/

Your exploration lives here. Week 2 is an EDA notebook: start from [`week-02-eda.ipynb`](week-02-eda.ipynb) and work through the eight `TODO`s.

```bash
pip install -r requirements-dev.txt     # seaborn and jupyter live here
jupyter notebook notebooks/week-02-eda.ipynb
```

(or)


You can open up the notebook in VScode if you install the plugin for it: https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter. This is probably easier and you can work on the notebook from the editor.

The data is gitignored, so the first cell regenerates it if it is missing — at whatever `STUDENT_SEED` is set in your environment. Check the seed the notebook prints in section 0 before you trust a figure.

Notebooks are for looking at things. Anything you need to *run again* — feature engineering, training, evaluation — belongs in the `.py` files at the repo root, which is where the tests can reach it.

Commit your notebooks. `.ipynb_checkpoints/` is already gitignored.

Two habits worth having from the start:

- **Restart and run all before you commit.** A notebook that only works in the order you happened to click is not a result anyone can reproduce. *Kernel → Restart & Run All.*
- **Write the sentence under the figure.** Every `______` in the starter is there because a plot nobody interpreted is not a finding Those sentences will frame what you present in Week 8.

Write your assigned seed here so you don't lose it:

    STUDENT_SEED =
