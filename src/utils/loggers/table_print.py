# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
"""
    Methods used to print a cute table for metrics
"""

import pandas as pd


def type_header(table, curr_type, col_width, tot_width, tot_col):
    # metric type header
    table += "\u2551" + \
        f"{f'{curr_type.upper()} METRICS':^{tot_width}}"+"\u2551"
    table += "\n"

    table += "\u2560"
    table += "\u2550"*(col_width)
    for i in range(tot_col-1):
        table += "\u2566"
        table += "\u2550"*(col_width-1)
    table += "\u2563"
    table += "\n"
    return table


def mid_bar(table, col_width, tot_col):
    table += "\u2560"
    table += "\u2550"*(col_width)
    for i in range(tot_col-1):
        table += "\u256C"
        table += "\u2550"*(col_width-1)
    table += "\u2563"
    table += "\n"
    return table


def metric_header(table, df: pd.DataFrame, col_width, tot_col):
    if ('epoch' in df.columns):
        ep = int(df['epoch'][0])
    else:
        ep = ""
    table += "\u2551"
    content = f"class \\ epoch {ep}"
    table += f"{content:^{col_width}}"
    for col_name in df.columns:
        if col_name != "epoch" and col_name != "class":
            table += "\u2551"
            table += f"{col_name:^{col_width-1}}"
    table += "\u2551"
    table += "\n"
    table = mid_bar(table, col_width=col_width, tot_col=tot_col)
    return table


def class_metrics(table, df: pd.DataFrame, col_width, tot_col):
    # table header
    for idx, row in df.iterrows():
        table += "\u2551"
        cls = f"class {int(row['class'])}"
        table += f"{cls:^{col_width}}"
        table += "\u2551"
        for id, val in row.items():
            if id != "epoch" and id != "class":
                value = f"{val:.3%}"
                table += f"{value:^{col_width-1}}"
                table += "\u2551"
        table += "\n"
        if (idx < len(df)-1):
            table = mid_bar(table, col_width=col_width, tot_col=tot_col)
    return table


def to_table(curr_type, df: pd.DataFrame, odd: bool, decim_digits: int) -> str:
    """
    Prints metrics as a table.

    Args:
        curr_type: The type of metrics.
        df: The DataFrame containing the metrics.
        odd: A boolean indicating whether the number of columns is odd.
        decim_digits: The number of decimal digits.

    Returns:
        str: The formatted table.
    """
    """ 
        \u2550 = '═'
        \u2551 = '║'
        \u2554 = '╔'
        \u2557 = '╗'
        \u255A = '╚'
        \u255D = '╝'
        \u2560 = '╠'
        \u2563 = '╣'
        \u2566 = '╦'
        \u2569 = '╩'
        \u256C = '╬'
    """
    col_width = 20
    if ('epoch' in df.columns):
        tot_col = len(df.columns) - 1
    else:
        tot_col = len(df.columns)
    tot_width = col_width * tot_col

    table = "\n"
    table += "\u2554"+"\u2550"*tot_width+"\u2557\n"
    table = type_header(table, curr_type, col_width=col_width,
                        tot_col=tot_col, tot_width=tot_width)
    table = metric_header(table, df, col_width, tot_col)
    table = class_metrics(table, df, col_width, tot_col)
    table += "\u255A"+"\u2550"*tot_width+"\u255D\n"
    return table
