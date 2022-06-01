import datetime
import os

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
import seaborn as sns

TEAM_COLORS = {
    "Mercedes":"#00D2BE",
    "Custom":(22.6/255.0, 21.1/255.0, 29.0/255.0),
    "Mc Laren":"#FF8700",
    "Ferrari":"#DC0000",
    "Red Bull":"#0600EF",
    "Alpine":"#0090FF",
    "Aston Martin":"#006F62",
    "Haas":"#8c8c8c",
    "Williams":"#005AFF",
    "Alfa Romeo":"#900000",
    "AlphaTauri":"#2B4562",
}

def check_date(datestr):
    try:
        datetime.datetime.strptime(datestr, "%d/%m/%Y")
        return True
    except:
        return False


def read_and_filter_csv(filepath):
    df = pd.read_csv(filepath)

    # Filter mistakes with date and time
    df = df[(df.time.str.count(":") == 1) & (df.time.str.count("\\.") == 1) & (df.date.str.count("/") == 2)]
    df = df[df.time.str.len() == 8]
    df = df[(df.custom == "Yes") | (df.custom == "No")]
    df = df[df["time"].str.split(".").str[1].str.len() == 3]
    df = df[df["time"].str.split(".").str[0].str.len() == 4]

    # Time to seconds
    df["time_s"] = df.apply(lambda x: 60 * int(x["time"].split(":")[0]) + int(x["time"].split(":")[1].split(".")[0]) + 0.001 * int(x["time"].split(".")[1]), axis=1)

    # Filter outliers in time caused by mistakes
    df = df[df["time_s"] <= df.iloc[-1,:].time_s]

    # Sort by lap time and date
    df = df.sort_values(by=["time_s", "date"], ascending=[True, False]).reset_index(drop=True)

    # Filters out dates with wrong formats '/x/xxxx', 'x//xxxx', or 'x/x/' or any of their combination
    # correct date formatting for easy sorting
    df = df[df.date.apply(check_date)]
    pd.options.mode.chained_assignment = None
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")

    # Finally remove duplicate rows if left and reset index
    df = df.drop_duplicates(subset=["name","time"]).reset_index(drop=True)
    
    return df

def plot_time_dist(df, figures_path=None):
    plt.show() # fix weird bug where matplotlib/seaborn does not use settings for first plot
    rcParams['figure.figsize'] = 11.7,8.27
    sns.set_theme(style="darkgrid")
    sns.set(font_scale=1.2)

    ax = sns.histplot(x="time_s", data=df, kde=True, color="b", element="step", stat="percent")
    ax.set(xlabel="Time (s)", ylabel="Time Proportion (%)")
    ax.set_title(f"Time Distribution")

    if figures_path != None:
        plt.savefig(os.path.join(figures_path, "time_dist.png"), bbox_inches="tight")
    
    return plt.show()

def plot_cum_time_dist(df, figures_path=None):
    plt.show()
    rcParams['figure.figsize'] = 11.7,8.27
    sns.set_theme(style="darkgrid")
    sns.set(font_scale=1.2)

    ax = sns.histplot(x="time_s", data=df, kde=True, color="b", element="step", stat="percent", cumulative=True)
    ax.set(xlabel="Time (s)", ylabel="Time Proportion (%)")
    ax.set_title(f"Time Distribution (cumulative)")

    if figures_path != None:
        plt.savefig(os.path.join(figures_path, "time_cumulative_dist.png"), bbox_inches="tight")

    return plt.show()

def plot_team_dist(df, figures_path=None, window_width=5000):
    team_proportions = pd.get_dummies(df.team).rolling(window_width, center=True).sum() / window_width * 100.0

    rcParams['figure.figsize'] = 11.7,8.27
    sns.set_theme(style="darkgrid")
    sns.set(font_scale=1.2)

    ax = sns.lineplot(data=team_proportions, palette=TEAM_COLORS, dashes=False, markers=False, linewidth=1.0)
    ax.set(xlabel="Leaderboard Position", ylabel="Team Proportion (%)")
    ax.set_title(f"Team Distribution w.r.t. Leaderboard Position (rolling window size: {window_width})")

    team_labels = []
    team_counts = df.groupby("team").count().iloc[:,0]
    team_sum = team_counts.sum()
    teams_sorted = sorted(TEAM_COLORS.keys(), key= lambda x: team_counts[x], reverse=True)
    for team in teams_sorted:
        team_labels.append(f"{team} ({team_counts[team] * 100.0 / team_sum:.1f} %)")

    leg = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., labels=team_labels)

    for i, legobj in enumerate(leg.legendHandles):
        legobj.set_color(TEAM_COLORS[teams_sorted[i]])
        legobj.set_linewidth(2.0)

    ax.set_ylim([0,50])
    if figures_path != None:
        plt.savefig(os.path.join(figures_path, "team_dist.png"), dpi=100, bbox_inches="tight")
    return plt.show()


def plot_assist_dist(df, figures_path=None, window_width=5000):
    rcParams['figure.figsize'] = 11.7,8.27
    sns.set_theme(style="darkgrid")
    sns.set(font_scale=1.2)

    tc = (pd.get_dummies(df.tc).rolling(window_width, center=True).sum() / window_width * 100.0)[1]
    gear = (pd.get_dummies(df.gear).rolling(window_width, center=True).sum() / window_width * 100.0)[1]
    brake = (pd.get_dummies(df.brakes).rolling(window_width, center=True).sum() / window_width * 100.0)[1]
    custom = (pd.get_dummies(df.custom).rolling(window_width, center=True).sum() / window_width * 100.0)["Yes"]

    proportions = pd.concat([tc,gear,brake,custom], axis=1)
    proportions.columns = ["Traction Control", "Automatic", "ABS", "Custom Setup"]

    ax = sns.lineplot(data=proportions, dashes=False, markers=False, linewidth=1.5)
    ax.set(xlabel="Leaderboard Position", ylabel="Assist Proportion (%)")
    ax.set_title(f"Assist Usage Distribution w.r.t. Leaderboard Position (rolling window size: {window_width})")

    ax.set_ylim([0,100])

    leg = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)

    assist_labels = [
        f"Traction Control ({df.tc.sum() * 100.0 / df.shape[0]:.1f} %)",
        f"Automatic Gearbox ({df.gear.sum() * 100.0 / df.shape[0]:.1f} %)",
        f"ABS ({df.brakes.sum() * 100.0 / df.shape[0]:.1f} %)",
        f"Custom Setup ({df.groupby('custom').count().iloc[1,0] * 100.0 / df.shape[0]:.1f} %)",
    ]


    leg = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., labels=assist_labels)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)

    if figures_path != None:
        plt.savefig(os.path.join(figures_path, "assist_dist.png"), dpi=100, bbox_inches="tight")
    return plt.show()

def plot_weekday(df, circuit_name, gp_weeks=[], figures_path=None):
    rcParams['figure.figsize'] = 11.7,8.27
    sns.set_theme(style="darkgrid")
    sns.set(font_scale=1.2)

    for gp_week in gp_weeks:
        plt.axvline(gp_week-0.5, color="r")
        plt.text(gp_week-1.5, 3,f"{circuit_name.capitalize()} GP", rotation=90)


    df["week_num"] = df.date.apply(lambda x:  x.isocalendar()[1])
    ax = sns.histplot(x="week_num", data=df, bins=52, element="step", stat="percent")
    ax.set(xlabel="Week Number", ylabel="Date Proportion (%)")
    ax.set_title(f"Laptime Date Distribution")

    if figures_path != None:
        plt.savefig(os.path.join(figures_path, "weekday_dist.png"), dpi=100, bbox_inches="tight")
    return plt.show()

def generate_html(circuit_name, circuit_country, circuit_city):
    template_str = ""
    with open(os.path.join("..", "circuits", "template.html"), "r") as template:
        template_str = template.read()
    
    template_str = template_str.replace("$country$", circuit_country.capitalize())
    template_str = template_str.replace("$name$", circuit_name)
    template_str = template_str.replace("$Name$", circuit_name.capitalize())
    template_str = template_str.replace("$city$", circuit_city.capitalize())

    template_lines = template_str.split("\n")
    for i, line in enumerate(template_lines):
        if f'href="{circuit_name}.html"' in line:
            template_lines[i] = line.replace("nav-link", "nav-link active")
            break
    

    with open(os.path.join("..", "circuits", f"{circuit_name}.html"), "w") as out:
        out.write("\n".join(template_lines))

def plot_team_times(df, figures_path=None):
    rcParams['figure.figsize'] = 11.7,8.27
    sns.set_theme(style="darkgrid")
    sns.set(font_scale=1.2)

    avg_times = df.groupby("team").mean().reset_index().sort_values("time_s")
    avg_times["time_s"] = avg_times["time_s"] - avg_times["time_s"].iloc[0]
    ax = sns.barplot(x="team", y="time_s", data=avg_times, palette=TEAM_COLORS)

    ax.set(xlabel="Team", ylabel="Time Delta (s)")
    
    plt.xticks(rotation=45)
    ax.set_title(f"Average lap times by team")
    ax.bar_label(ax.containers[0], fmt="%.3f")

    if figures_path != None:
        plt.savefig(os.path.join(figures_path, "team_times.png"), dpi=100, bbox_inches="tight")
    
    return plt.show()