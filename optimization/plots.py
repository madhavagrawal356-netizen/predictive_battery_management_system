'''This module makes all the necessary plots to be displayed.'''
import matplotlib.pyplot as plt

# %%
def plot_soh_curve(df, battery_id):
    battery = df[df["Battery_ID"]==battery_id]
    fig,ax = plt.subplots(figsize=(8,5))

    ax.plot(
        battery["Cycle_Number"],
        battery["Predicted_SOH"],
        marker='x',
        linestyle='--',
        label="Predicted SOH"
    )

    ax.set_title(f"SOH Degradation - {battery_id}")
    ax.set_xlabel("Cycle Number")
    ax.set_ylabel("SOH")
    ax.grid(True)
    ax.legend()

    return fig

# %%
def plot_rul(df):
    data=df.sort_values('Remaining_useful_life')
    fig,ax = plt.subplots(figsize=(8,5))
    ax.barh(
        data["Battery_ID"],
        data["Remaining_useful_life"]
    )

    ax.set_xlabel("Remaining Useful Life (Cycles)")
    ax.set_ylabel("Battery")
    ax.set_title("Remaining Useful Life")

    return fig


# %%
def plot_risk(df):
    data = df.sort_values("Risk")
    fig, ax = plt.subplots(figsize=(8,5))
    ax.barh(
        data["Battery_ID"],
        data["Risk"])
    ax.set_xlabel("Risk Score")
    ax.set_ylabel("Battery")
    ax.set_title("Battery Risk Scores")
    return fig


