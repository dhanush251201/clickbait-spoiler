"""Small helpers for writing result tables as markdown (no extra dependency such as tabulate)."""
import math


def to_markdown(df, digits=4):
    """A plain markdown table of a pandas DataFrame: floats with `digits` decimals, missing values left blank."""
    def fmt(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return ""
        if isinstance(v, float):
            return f"{v:.{digits}f}"
        return str(v).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(str(c) for c in df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"
