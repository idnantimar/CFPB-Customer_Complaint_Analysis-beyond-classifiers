from .nlp_spaCy import spaCy_cleaner
from .stability_analysis_helper import split_OddEvenRand
from . import find_aliasCo
from . import embed_topics
import pandas as pd
from IPython.display import display,HTML


__all__ = [
    "spaCy_cleaner",
    "split_OddEvenRand",
    "find_aliasCo",
    "embed_topics",
    "format_df",
]


def format_df(df:pd.DataFrame,*,k:int=2,heading:str|None=None) :
    """
        Instead of saving rounded-off data, only do it at display time.
        Add an heading (optional).
    """
    if heading is not None : display(HTML(f"""<h2 style="margin-top:0.4em; margin-bottom:0.1em;"><u>{heading}</u></h2>"""))
    display(
        df.style
        .format({
            **{
                col: f"{{:.{k}f}}"
                for col in df.select_dtypes(include='floating').columns
            },       
        })
        .set_table_styles([
            {
                "selector": "th, td",
                "props": [
                    ("border-right", '1px solid #bbb'),
                    ("border-bottom", '1px dashed #999'),
                  ],
            },
            {
                "selector": ".row_heading",
                "props": [
                    ("font-size", '110%'),
                    ("font-weight", 'bold'),
                ],
            },
            {
                "selector": ".col_heading",
                "props": [
                    ("font-size", '115%'),
                    ("font-weight", 'bold'),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("white-space", 'normal'),
                    ("word-break", 'break-word'),
                    ("overflow-wrap", 'break-word'),
                ]
            },
        ])
        .set_properties(
            subset=df.select_dtypes(include=['string','object']).columns,
            **{
                "text-align": 'center',
                "max-width": '1500px',
            }
        )
    )